"""
Phase 4 Experiment Script - Final targeted improvements and diagnostics.

Experiments:
A: Learnable initial state (h0)
B: Attention + AdamW
C: Attention + h0 (if A helps)
D: Attention + learnable alpha (optional)

Usage:
    python scripts/experiment_phase4.py --data data/rnn_export_Newton_08_15_2025_SC.mat --output results/phase4/
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
    compute_L_neuron, compute_L_trial, compute_L_reg,
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


def compute_per_neuron_correlations(
    model: EIRNN,
    data: dict,
    device: str
) -> np.ndarray:
    """Compute PSTH correlation for each neuron."""
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
        correlations = np.zeros(n_recorded)
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            correlations[i] = r if not np.isnan(r) else 0.0

        return correlations


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
    """Train a model with the given configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=bin_size_ms,
        noise_scale=config.get('noise_scale', 0.1),
        learnable_tau=config.get('learnable_tau', 'none'),
        tau_e_init=config.get('tau_e_init', 50.0),
        tau_i_init=config.get('tau_i_init', 35.0),
        learnable_alpha=config.get('learnable_alpha', 'none'),
        alpha_init=config.get('alpha_init', 0.5),
        alpha_e_init=config.get('alpha_e_init', None),
        alpha_i_init=config.get('alpha_i_init', None),
        input_embed_dim=config.get('input_embed_dim', None),
        input_embed_type=config.get('input_embed_type', 'learnable'),
        attention_heads=config.get('attention_heads', 4),
        learnable_h0=config.get('learnable_h0', False),
        h0_init=config.get('h0_init', 0.1),
        device=device
    )

    # Print model info
    name = config.get('name', 'unknown')
    print(f"  [{name}] Attention embedding: dim={config.get('input_embed_dim')}, heads={config.get('attention_heads', 4)}")
    if config.get('learnable_h0', False):
        print(f"  [{name}] Learnable initial state: h0_init={config.get('h0_init', 0.1)}")
    if config.get('learnable_alpha', 'none') != 'none':
        print(f"  [{name}] Learnable alpha: mode={config['learnable_alpha']}, E_init={config.get('alpha_e_init')}, I_init={config.get('alpha_i_init')}")

    # Optimizer
    lr = config.get('lr', 1e-3)
    optimizer_type = config.get('optimizer', 'adam')

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'adamw':
        # AdamW with weight decay on embedding only
        embed_params = []
        other_params = []
        for name_p, param in model.named_parameters():
            if 'input_embed' in name_p:
                embed_params.append(param)
            else:
                other_params.append(param)

        weight_decay = config.get('weight_decay', 0.01)
        optimizer = torch.optim.AdamW([
            {'params': embed_params, 'weight_decay': weight_decay},
            {'params': other_params, 'weight_decay': 0.0}
        ], lr=lr)
        print(f"  [{name}] Using AdamW with weight_decay={weight_decay} on embedding")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    # Loss configuration
    use_grad_balancing = config.get('use_grad_balancing', True)
    ltrial_scale = config.get('ltrial_scale', 0.5)
    lambda_reg = config.get('lambda_reg', 1e-4)

    # For gradient balancing, track EMAs
    loss_ema = {'L_neuron': 1.0, 'L_trial': 1.0}
    ema_decay = 0.99

    # Training loop
    max_epochs = config.get('max_epochs', 500)
    patience = config.get('patience', 100)
    best_val_corr = float('-inf')
    epochs_without_improvement = 0
    best_model_state = None

    history = {
        'train_corr': [],
        'val_corr': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'alpha_values': [],
        'h0_values': []
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

        # Compute losses
        L_neuron = compute_L_neuron(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, lambda_scale=0.1, lambda_var=0.05
        )

        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, sinkhorn_iters=20, sinkhorn_epsilon=0.1
        )

        L_reg = compute_L_reg(model, lambda_reg)

        # Gradient balancing
        if use_grad_balancing:
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial_raw.item()

            L_neuron_norm = L_neuron / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial_raw / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + ltrial_scale * L_trial_norm + L_reg
        else:
            loss = L_neuron + 0.5 * L_trial_raw + L_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        train_corr = compute_psth_correlation(model, train_data, device)

        # Validate
        model.eval()
        with torch.no_grad():
            inputs_val = val_data['inputs'].to(device)
            targets_val = val_data['targets'].to(device)
            mask_val = val_data['mask'].to(device)

            model_rates_val, _ = model(inputs_val)
            model_rates_val_recorded = model_rates_val[:, :, :n_recorded]

            L_neuron_val = compute_L_neuron(
                model_rates_val_recorded, targets_val, bin_size_ms,
                mask=mask_val, lambda_scale=0.1, lambda_var=0.05
            )

            val_loss_value = L_neuron_val.item()
            val_corr = compute_psth_correlation(model, val_data, device)

        # Update scheduler
        scheduler.step(val_loss_value)

        # Record history
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_value)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Record alpha/h0 values if learnable
        if hasattr(model, 'get_alpha_values'):
            history['alpha_values'].append(model.get_alpha_values())
        if model.h0 is not None:
            h0_vals = model.h0.detach().cpu().numpy()
            history['h0_values'].append({
                'mean': float(h0_vals.mean()),
                'std': float(h0_vals.std()),
                'min': float(h0_vals.min()),
                'max': float(h0_vals.max())
            })

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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Get final values
    alpha_values = model.get_alpha_values() if hasattr(model, 'get_alpha_values') else {}
    h0_values = {}
    if model.h0 is not None:
        h0_vals = model.h0.detach().cpu().numpy()
        h0_values = {
            'mean': float(h0_vals.mean()),
            'std': float(h0_vals.std()),
            'min': float(h0_vals.min()),
            'max': float(h0_vals.max())
        }

    return {
        'best_val_corr': best_val_corr,
        'final_val_corr': val_corr,
        'history': history,
        'epochs_trained': epoch + 1,
        'alpha_values': alpha_values,
        'h0_values': h0_values,
        'config': config,
        'best_model_state': best_model_state
    }


def run_experiments(
    data_path: str,
    output_dir: str,
    device: str = 'cpu',
    seed: int = 42,
    experiments_to_run: list = None
):
    """Run Phase 4 experiments."""
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
    print(f"Input dim: {n_inputs}, Bin size: {bin_size_ms}ms")

    # Phase 3 best baseline for reference
    phase3_best = 0.3843

    # Define experiments
    all_experiments = {
        # Experiment A: Attention + Learnable h0
        'attention_learnable_h0': {
            'name': 'attention_learnable_h0',
            'input_embed_dim': 56,
            'input_embed_type': 'attention',
            'attention_heads': 4,
            'learnable_h0': True,
            'h0_init': 0.1,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment B: Attention + AdamW
        'attention_adamw': {
            'name': 'attention_adamw',
            'input_embed_dim': 56,
            'input_embed_type': 'attention',
            'attention_heads': 4,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment C: Attention + h0 + AdamW (combine A and B)
        'attention_h0_adamw': {
            'name': 'attention_h0_adamw',
            'input_embed_dim': 56,
            'input_embed_type': 'attention',
            'attention_heads': 4,
            'learnable_h0': True,
            'h0_init': 0.1,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Experiment D: Attention + Learnable alpha
        'attention_alpha_neuron': {
            'name': 'attention_alpha_neuron',
            'input_embed_dim': 56,
            'input_embed_type': 'attention',
            'attention_heads': 4,
            'learnable_alpha': 'neuron',
            'alpha_e_init': 0.54,
            'alpha_i_init': 0.72,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Baseline: Attention only (for comparison)
        'attention_baseline': {
            'name': 'attention_baseline',
            'input_embed_dim': 56,
            'input_embed_type': 'attention',
            'attention_heads': 4,
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },
    }

    # Filter experiments if specified
    if experiments_to_run:
        experiments = {k: v for k, v in all_experiments.items() if k in experiments_to_run}
    else:
        experiments = all_experiments

    # Run experiments
    results = {}
    for name, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}")

        try:
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

            results[name] = result

            # Save intermediate results
            result_path = output_dir / f"{name}_result.json"
            result_json = {
                'best_val_corr': result['best_val_corr'],
                'final_val_corr': result['final_val_corr'],
                'epochs_trained': result['epochs_trained'],
                'alpha_values': result['alpha_values'],
                'h0_values': result['h0_values'],
                'config': result['config']
            }
            with open(result_path, 'w') as f:
                json.dump(result_json, f, indent=2)

            # Save best model
            if result.get('best_model_state') is not None:
                model_path = output_dir / f"{name}_model_best.pt"
                torch.save(result['best_model_state'], model_path)

            print(f"Best val correlation: {result['best_val_corr']:.4f}")

        except Exception as e:
            print(f"ERROR in experiment {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'best_val_corr': float('-inf'), 'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("PHASE 4 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Configuration':<35} {'Best Val Corr':>15}")
    print("-"*50)

    target = 0.40
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1]['best_val_corr'],
        reverse=True
    )
    for name, result in sorted_results:
        marker = ""
        if result['best_val_corr'] >= target:
            marker = " ** TARGET **"
        elif result['best_val_corr'] > phase3_best:
            marker = " *"
        print(f"{name:<35} {result['best_val_corr']:>15.4f}{marker}")

    print("\n" + "-"*50)
    print("Reference points:")
    print(f"  Phase 3 best (attention):          {phase3_best:.4f}")
    print(f"  Target:                            {target:.4f}")

    if sorted_results:
        best_name, best_result = sorted_results[0]
        improvement = (best_result['best_val_corr'] - phase3_best) / phase3_best * 100
        print(f"\nPhase 4 best ({best_name}):")
        print(f"  Best val corr: {best_result['best_val_corr']:.4f}")
        print(f"  vs Phase 3:    {improvement:+.1f}%")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'phase3_best': phase3_best,
        'target': target,
        'results': {
            name: {
                'best_val_corr': r.get('best_val_corr', float('-inf')),
                'final_val_corr': r.get('final_val_corr'),
                'epochs_trained': r.get('epochs_trained'),
                'alpha_values': r.get('alpha_values', {}),
                'h0_values': r.get('h0_values', {}),
                'error': r.get('error')
            }
            for name, r in results.items()
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot comparison
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        plot_comparison(valid_results, output_dir / 'comparison.png', phase3_best=phase3_best, target=target)

    return results


def plot_comparison(results: dict, save_path: str, phase3_best: float = 0.3843, target: float = 0.40):
    """Plot comparison of experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax = axes[0]
    names = list(results.keys())
    corrs = [results[n]['best_val_corr'] for n in names]

    sorted_idx = np.argsort(corrs)[::-1]
    names = [names[i] for i in sorted_idx]
    corrs = [corrs[i] for i in sorted_idx]

    colors = ['green' if c >= target else 'limegreen' if c > phase3_best else 'steelblue' for c in corrs]

    bars = ax.barh(names, corrs, color=colors)
    ax.set_xlabel('Best Validation PSTH Correlation')
    ax.set_title('Phase 4 Experiment Comparison')
    ax.axvline(x=target, color='green', linestyle='--', label=f'Target ({target})')
    ax.axvline(x=phase3_best, color='orange', linestyle='--', label=f'Phase 3 Best ({phase3_best})')
    ax.legend(fontsize=8)
    ax.set_xlim(min(min(corrs) - 0.02, 0.35), max(max(corrs) + 0.02, target + 0.02))

    # Training curves
    ax = axes[1]
    for name in names[:5]:
        if 'history' in results[name]:
            history = results[name]['history']
            ax.plot(history['val_corr'], label=f"{name} ({results[name]['best_val_corr']:.3f})")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('Training Curves')
    ax.axhline(y=target, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=phase3_best, color='orange', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 4 experiments')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results/phase4/', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to run (default: all)')

    args = parser.parse_args()

    run_experiments(
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        seed=args.seed,
        experiments_to_run=args.experiments
    )


if __name__ == "__main__":
    main()
