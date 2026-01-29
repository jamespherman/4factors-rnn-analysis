"""
Experiment script to test model improvements.

Tests:
1. Poisson loss variants
2. Learnable time constants
3. Cosine annealing scheduler

Usage:
    python scripts/experiment_improvements.py --data path/to/data.mat --output results/improvements/
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import (
    EIRNNLoss, compute_L_neuron, compute_L_poisson, compute_L_neuron_hybrid,
    compute_activity_regularization, compute_L_trial, compute_L_reg,
    get_cosine_schedule_with_warmup
)
from src.data_loader import load_session, train_val_split


def compute_psth_correlation(
    model: EIRNN,
    data: dict,
    device: str
) -> float:
    """Compute mean PSTH correlation across neurons."""
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Trial-average
        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        # Correlation per neuron
        n_recorded = target_psth.shape[1]
        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

        return np.mean(correlations) if correlations else 0.0


def train_with_config(
    config: dict,
    train_data: dict,
    val_data: dict,
    neuron_info: dict,
    n_inputs: int,
    bin_size_ms: float,
    device: str = 'cpu',
    seed: int = 42
) -> dict:
    """
    Train a model with the given configuration.

    Args:
        config: Dict with training configuration
        train_data: Training data dict
        val_data: Validation data dict
        neuron_info: Dict with n_exc, n_inh
        n_inputs: Number of input dimensions
        bin_size_ms: Bin size in ms
        device: Device to use
        seed: Random seed

    Returns:
        Dict with training results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=bin_size_ms,
        learnable_tau=config.get('learnable_tau', 'none'),
        tau_e_init=config.get('tau_e_init', 50.0),
        tau_i_init=config.get('tau_i_init', 20.0),
        device=device
    )

    # Optimizer
    lr = config.get('lr', 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler_type = config.get('scheduler', 'plateau')
    max_epochs = config.get('max_epochs', 500)

    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
        )
    elif scheduler_type == 'cosine':
        warmup_epochs = config.get('warmup_epochs', 50)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_epochs, max_epochs
        )
    else:
        scheduler = None

    # Loss configuration
    loss_type = config.get('loss_type', 'correlation')  # 'correlation', 'poisson', 'hybrid'
    poisson_weight = config.get('poisson_weight', 0.5)
    use_activity_reg = config.get('use_activity_reg', False)
    lambda_activity = config.get('lambda_activity', 0.01)
    lambda_trial = config.get('lambda_trial', 0.5)
    lambda_reg = config.get('lambda_reg', 1e-4)

    # Gradient balancing parameters
    use_grad_balancing = config.get('use_grad_balancing', True)
    ltrial_scale = config.get('ltrial_scale', 0.5)

    # For gradient balancing, track EMAs
    loss_ema = {'L_neuron': 1.0, 'L_trial': 1.0}
    ema_decay = 0.99

    # Training loop
    patience = config.get('patience', 100)
    best_val_corr = float('-inf')
    epochs_without_improvement = 0

    history = {
        'train_corr': [],
        'val_corr': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    pbar = tqdm(range(max_epochs), desc=config.get('name', 'Training'))
    for epoch in pbar:
        # Train
        model.train()
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        optimizer.zero_grad()
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        model_rates_recorded = model_rates[:, :, :n_recorded]

        # Compute losses based on loss_type
        if loss_type == 'correlation':
            L_neuron = compute_L_neuron(
                model_rates_recorded, targets, bin_size_ms,
                mask=mask, lambda_scale=0.1, lambda_var=0.05
            )
        elif loss_type == 'poisson':
            L_neuron = compute_L_poisson(
                model_rates_recorded, targets, bin_size_ms, mask=mask
            )
        elif loss_type == 'hybrid':
            L_neuron = compute_L_neuron_hybrid(
                model_rates_recorded, targets, bin_size_ms,
                mask=mask, poisson_weight=poisson_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # L_trial with Sinkhorn
        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, sinkhorn_iters=20, sinkhorn_epsilon=0.1
        )

        # L_reg
        L_reg = compute_L_reg(model, lambda_reg)

        # Activity regularization
        if use_activity_reg:
            L_activity = compute_activity_regularization(
                model_rates_recorded, lambda_mean=lambda_activity
            )
        else:
            L_activity = torch.tensor(0.0, device=device)

        # Gradient balancing
        if use_grad_balancing:
            # Update EMAs
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial_raw.item()

            # Normalize and combine
            L_neuron_norm = L_neuron / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial_raw / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + ltrial_scale * L_trial_norm + L_reg + L_activity
        else:
            loss = L_neuron + lambda_trial * L_trial_raw + L_reg + L_activity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        train_corr = compute_psth_correlation(model, train_data, device)

        # Validate
        val_loss_value, val_corr = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            inputs_val = val_data['inputs'].to(device)
            targets_val = val_data['targets'].to(device)
            mask_val = val_data['mask'].to(device)

            model_rates_val, _ = model(inputs_val)
            model_rates_val_recorded = model_rates_val[:, :, :n_recorded]

            if loss_type == 'correlation':
                L_neuron_val = compute_L_neuron(
                    model_rates_val_recorded, targets_val, bin_size_ms,
                    mask=mask_val, lambda_scale=0.1, lambda_var=0.05
                )
            elif loss_type == 'poisson':
                L_neuron_val = compute_L_poisson(
                    model_rates_val_recorded, targets_val, bin_size_ms, mask=mask_val
                )
            elif loss_type == 'hybrid':
                L_neuron_val = compute_L_neuron_hybrid(
                    model_rates_val_recorded, targets_val, bin_size_ms,
                    mask=mask_val, poisson_weight=poisson_weight
                )

            val_loss_value = L_neuron_val.item()
            val_corr = compute_psth_correlation(model, val_data, device)

        # Update scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_loss_value)
        elif scheduler_type == 'cosine':
            scheduler.step()

        # Record history
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_value)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Update progress
        pbar.set_postfix({
            'train_corr': f'{train_corr:.3f}',
            'val_corr': f'{val_corr:.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })

        # Early stopping
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Get tau values if learnable
    tau_values = model.get_tau_values() if hasattr(model, 'get_tau_values') else {}

    return {
        'best_val_corr': best_val_corr,
        'final_val_corr': val_corr,
        'history': history,
        'epochs_trained': epoch + 1,
        'tau_values': tau_values,
        'config': config
    }


def run_experiments(
    data_path: str,
    output_dir: str,
    device: str = 'cpu',
    seed: int = 42
):
    """Run all improvement experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    dataset = load_session(data_path)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)

    all_data = dataset.get_all_trials()
    train_data = {
        'inputs': all_data['inputs'][train_idx],
        'targets': all_data['targets'][train_idx],
        'mask': all_data['mask'][train_idx],
    }
    val_data = {
        'inputs': all_data['inputs'][val_idx],
        'targets': all_data['targets'][val_idx],
        'mask': all_data['mask'][val_idx],
    }

    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()
    bin_size_ms = dataset.bin_size_ms

    print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")

    # Define experiment configurations
    experiments = [
        # Baseline (current best)
        {
            'name': 'baseline',
            'loss_type': 'correlation',
            'learnable_tau': 'none',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment 1: Loss function variants
        {
            'name': 'poisson_loss',
            'loss_type': 'poisson',
            'learnable_tau': 'none',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },
        {
            'name': 'hybrid_loss_0.3',
            'loss_type': 'hybrid',
            'poisson_weight': 0.3,
            'learnable_tau': 'none',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },
        {
            'name': 'hybrid_loss_0.5',
            'loss_type': 'hybrid',
            'poisson_weight': 0.5,
            'learnable_tau': 'none',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment 2: Learnable time constants
        {
            'name': 'learnable_tau_population',
            'loss_type': 'correlation',
            'learnable_tau': 'population',
            'tau_e_init': 50.0,
            'tau_i_init': 20.0,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },
        {
            'name': 'learnable_tau_neuron',
            'loss_type': 'correlation',
            'learnable_tau': 'neuron',
            'tau_e_init': 50.0,
            'tau_i_init': 20.0,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment 3: Learning rate schedules
        {
            'name': 'cosine_no_warmup',
            'loss_type': 'correlation',
            'learnable_tau': 'none',
            'scheduler': 'cosine',
            'warmup_epochs': 0,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },
        {
            'name': 'cosine_warmup_50',
            'loss_type': 'correlation',
            'learnable_tau': 'none',
            'scheduler': 'cosine',
            'warmup_epochs': 50,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment 4: Activity regularization
        {
            'name': 'activity_reg_0.01',
            'loss_type': 'correlation',
            'learnable_tau': 'none',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'use_activity_reg': True,
            'lambda_activity': 0.01,
            'max_epochs': 500,
            'patience': 100
        },
    ]

    # Run experiments
    results = {}
    for config in experiments:
        print(f"\n{'='*60}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*60}")

        result = train_with_config(
            config=config,
            train_data=train_data,
            val_data=val_data,
            neuron_info=neuron_info,
            n_inputs=n_inputs,
            bin_size_ms=bin_size_ms,
            device=device,
            seed=seed
        )

        results[config['name']] = result

        # Save intermediate results
        result_path = output_dir / f"{config['name']}_result.json"
        # Convert numpy arrays to lists for JSON serialization
        result_json = {
            'best_val_corr': result['best_val_corr'],
            'final_val_corr': result['final_val_corr'],
            'epochs_trained': result['epochs_trained'],
            'tau_values': result['tau_values'],
            'config': result['config']
        }
        with open(result_path, 'w') as f:
            json.dump(result_json, f, indent=2)

        print(f"Best val correlation: {result['best_val_corr']:.4f}")

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Configuration':<30} {'Best Val Corr':>15}")
    print("-"*45)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_corr'], reverse=True)
    for name, result in sorted_results:
        print(f"{name:<30} {result['best_val_corr']:>15.4f}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'results': {
            name: {
                'best_val_corr': r['best_val_corr'],
                'final_val_corr': r['final_val_corr'],
                'epochs_trained': r['epochs_trained'],
                'tau_values': r['tau_values']
            }
            for name, r in results.items()
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot comparison
    plot_comparison(results, output_dir / 'comparison.png')

    return results


def plot_comparison(results: dict, save_path: str):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of best correlations
    ax = axes[0]
    names = list(results.keys())
    corrs = [results[n]['best_val_corr'] for n in names]

    colors = ['green' if c == max(corrs) else 'steelblue' for c in corrs]
    bars = ax.barh(names, corrs, color=colors)
    ax.set_xlabel('Best Validation PSTH Correlation')
    ax.set_title('Experiment Comparison')
    ax.axvline(x=0.3614, color='red', linestyle='--', label='Current Best (0.3614)')
    ax.legend()

    # Training curves for top 3
    ax = axes[1]
    sorted_names = sorted(names, key=lambda n: results[n]['best_val_corr'], reverse=True)[:3]
    for name in sorted_names:
        history = results[name]['history']
        ax.plot(history['val_corr'], label=f"{name} ({results[name]['best_val_corr']:.3f})")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('Training Curves (Top 3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test model improvements')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results/improvements/',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    run_experiments(
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
