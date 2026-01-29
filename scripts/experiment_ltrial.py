"""
Experimental script for L_trial optimization strategies.

Tests multiple approaches to find optimal trial-matching configuration:
1. Different lambda_trial values
2. Different Sinkhorn epsilon values
3. Gradient balancing (Défossez et al. 2023)
4. Combinations
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import compute_L_neuron, compute_L_trial, compute_L_reg
from src.data_loader import load_session, train_val_split


def compute_psth_correlation(model, data, device):
    """Compute mean PSTH correlation across neurons."""
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)

        n_recorded = targets.shape[2]
        model_psth = model_rates[:, :, :n_recorded].mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

        return np.mean(correlations) if correlations else 0.0


def train_with_config(
    data_path: str,
    output_dir: str,
    config: dict,
    max_epochs: int = 500,
    patience: int = 100,
    device: str = 'cpu',
    seed: int = 42,
    verbose: bool = True
):
    """
    Train model with specified configuration.

    Config options:
        lambda_trial: Weight for L_trial (default 1.0)
        sinkhorn_epsilon: Entropy regularization for Sinkhorn (default 0.1)
        sinkhorn_iters: Number of Sinkhorn iterations (default 20)
        gradient_balancing: Whether to use gradient balancing (default False)
        lambda_scale: Weight for scale loss in L_neuron (default 0.1)
        lambda_var: Weight for variance loss in L_neuron (default 0.05)
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract config
    lambda_trial = config.get('lambda_trial', 1.0)
    sinkhorn_epsilon = config.get('sinkhorn_epsilon', 0.1)
    sinkhorn_iters = config.get('sinkhorn_iters', 20)
    gradient_balancing = config.get('gradient_balancing', False)
    lambda_scale = config.get('lambda_scale', 0.1)
    lambda_var = config.get('lambda_var', 0.05)
    lambda_reg = config.get('lambda_reg', 1e-4)

    # Load data
    dataset = load_session(data_path, validate=verbose)
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

    # Create model
    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()

    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        dt=float(dataset.bin_size_ms),
        device=device
    )

    bin_size_ms = dataset.bin_size_ms

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5
    )

    # Training loop
    history = {
        'train_L_neuron': [], 'val_L_neuron': [],
        'train_L_trial': [], 'val_L_trial': [],
        'train_psth_corr': [], 'val_psth_corr': [],
    }

    best_val_corr = float('-inf')
    epochs_without_improvement = 0

    pbar = tqdm(range(max_epochs), desc=f"λ={lambda_trial},ε={sinkhorn_epsilon}", disable=not verbose)

    for epoch in pbar:
        model.train()

        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        # Forward pass
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]

        # Compute losses
        L_neuron = compute_L_neuron(
            model_rates[:, :, :n_recorded], targets,
            bin_size_ms=bin_size_ms, mask=mask,
            lambda_scale=lambda_scale, lambda_var=lambda_var
        )

        if lambda_trial > 0:
            L_trial = compute_L_trial(
                model_rates[:, :, :n_recorded], targets,
                bin_size_ms=bin_size_ms, mask=mask,
                sinkhorn_iters=sinkhorn_iters, sinkhorn_epsilon=sinkhorn_epsilon
            )
        else:
            L_trial = torch.tensor(0.0, device=device)

        L_reg = compute_L_reg(model, lambda_l2=lambda_reg)

        # Backward pass
        optimizer.zero_grad()

        if gradient_balancing and lambda_trial > 0:
            # Gradient balancing: normalize gradients from each loss

            # Get gradient norm for L_neuron
            L_neuron.backward(retain_graph=True)
            grad_norm_neuron = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            # Get gradient norm for L_trial
            optimizer.zero_grad()
            L_trial.backward(retain_graph=True)
            grad_norm_trial = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            # Compute balanced loss
            optimizer.zero_grad()
            balanced_loss = (
                L_neuron / (grad_norm_neuron + 1e-8) +
                L_trial / (grad_norm_trial + 1e-8) +
                L_reg
            )
            balanced_loss.backward()
        else:
            # Standard weighted loss
            total_loss = L_neuron + lambda_trial * L_trial + L_reg
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        train_psth_corr = compute_psth_correlation(model, train_data, device)
        val_psth_corr = compute_psth_correlation(model, val_data, device)

        # Validation losses
        model.eval()
        with torch.no_grad():
            val_inputs = val_data['inputs'].to(device)
            val_targets = val_data['targets'].to(device)
            val_mask = val_data['mask'].to(device)

            val_rates, _ = model(val_inputs)
            val_L_neuron = compute_L_neuron(
                val_rates[:, :, :n_recorded], val_targets,
                bin_size_ms=bin_size_ms, mask=val_mask,
                lambda_scale=lambda_scale, lambda_var=lambda_var
            ).item()

            if lambda_trial > 0:
                val_L_trial = compute_L_trial(
                    val_rates[:, :, :n_recorded], val_targets,
                    bin_size_ms=bin_size_ms, mask=val_mask,
                    sinkhorn_iters=sinkhorn_iters, sinkhorn_epsilon=sinkhorn_epsilon
                ).item()
            else:
                val_L_trial = 0.0

        scheduler.step(val_psth_corr)

        # Record history
        history['train_L_neuron'].append(L_neuron.item())
        history['val_L_neuron'].append(val_L_neuron)
        history['train_L_trial'].append(L_trial.item() if isinstance(L_trial, torch.Tensor) else L_trial)
        history['val_L_trial'].append(val_L_trial)
        history['train_psth_corr'].append(train_psth_corr)
        history['val_psth_corr'].append(val_psth_corr)

        # Update progress
        pbar.set_postfix({
            'corr': f"{val_psth_corr:.3f}",
            'L_n': f"{L_neuron.item():.3f}",
            'L_t': f"{L_trial.item() if isinstance(L_trial, torch.Tensor) else 0:.2f}",
        })

        # Check for improvement
        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_psth_corr': val_psth_corr,
                'config': config,
            }, str(output_dir / 'model_best.pt'))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save results
    results = {
        'config': config,
        'best_val_corr': best_val_corr,
        'final_val_corr': val_psth_corr,
        'final_epoch': epoch,
        'history': history,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return best_val_corr, results


def run_experiment_grid(data_path: str, base_output_dir: str, device: str = 'cpu'):
    """Run grid of experiments with different configurations."""

    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Define experiment configurations
    experiments = [
        # Baseline: L_neuron only
        {'name': 'baseline_no_ltrial', 'lambda_trial': 0.0},

        # Different lambda_trial values with default Sinkhorn
        {'name': 'lambda_0.1_eps_0.1', 'lambda_trial': 0.1, 'sinkhorn_epsilon': 0.1},
        {'name': 'lambda_0.3_eps_0.1', 'lambda_trial': 0.3, 'sinkhorn_epsilon': 0.1},
        {'name': 'lambda_0.5_eps_0.1', 'lambda_trial': 0.5, 'sinkhorn_epsilon': 0.1},
        {'name': 'lambda_1.0_eps_0.1', 'lambda_trial': 1.0, 'sinkhorn_epsilon': 0.1},

        # Higher Sinkhorn epsilon (softer assignment)
        {'name': 'lambda_0.3_eps_0.5', 'lambda_trial': 0.3, 'sinkhorn_epsilon': 0.5},
        {'name': 'lambda_0.3_eps_1.0', 'lambda_trial': 0.3, 'sinkhorn_epsilon': 1.0},
        {'name': 'lambda_1.0_eps_0.5', 'lambda_trial': 1.0, 'sinkhorn_epsilon': 0.5},
        {'name': 'lambda_1.0_eps_1.0', 'lambda_trial': 1.0, 'sinkhorn_epsilon': 1.0},

        # Gradient balancing
        {'name': 'gradbal_eps_0.1', 'lambda_trial': 1.0, 'sinkhorn_epsilon': 0.1, 'gradient_balancing': True},
        {'name': 'gradbal_eps_0.5', 'lambda_trial': 1.0, 'sinkhorn_epsilon': 0.5, 'gradient_balancing': True},
        {'name': 'gradbal_eps_1.0', 'lambda_trial': 1.0, 'sinkhorn_epsilon': 1.0, 'gradient_balancing': True},
    ]

    results_summary = []

    print("="*70)
    print("EXPERIMENT GRID: L_trial Optimization")
    print("="*70)
    print(f"Running {len(experiments)} experiments...")
    print()

    for i, exp in enumerate(experiments):
        name = exp.pop('name')
        print(f"\n[{i+1}/{len(experiments)}] Running: {name}")
        print(f"  Config: {exp}")

        output_dir = base_output_dir / name

        try:
            best_corr, full_results = train_with_config(
                data_path=data_path,
                output_dir=str(output_dir),
                config=exp,
                max_epochs=500,
                patience=100,
                device=device,
                verbose=False
            )

            results_summary.append({
                'name': name,
                'config': exp,
                'best_val_corr': best_corr,
                'final_epoch': full_results['final_epoch'],
            })

            print(f"  Result: best_val_corr = {best_corr:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                'name': name,
                'config': exp,
                'best_val_corr': None,
                'error': str(e),
            })

        # Re-add name for next iteration reference
        exp['name'] = name

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Experiment':<30} {'Best Corr':>12} {'Config':<30}")
    print("-"*70)

    # Sort by correlation
    valid_results = [r for r in results_summary if r['best_val_corr'] is not None]
    valid_results.sort(key=lambda x: x['best_val_corr'], reverse=True)

    for r in valid_results:
        config_str = ', '.join([f"{k}={v}" for k, v in r['config'].items() if k != 'name'])
        print(f"{r['name']:<30} {r['best_val_corr']:>12.4f} {config_str:<30}")

    # Save summary
    with open(base_output_dir / 'summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {base_output_dir}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description='L_trial optimization experiments')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results/ltrial_experiments', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--single', type=str, default=None, help='Run single experiment by name')

    args = parser.parse_args()

    if args.single:
        # Run single experiment (for testing)
        config = {'lambda_trial': 0.3, 'sinkhorn_epsilon': 0.1}
        train_with_config(args.data, args.output, config, device=args.device)
    else:
        # Run full grid
        run_experiment_grid(args.data, args.output, args.device)


if __name__ == "__main__":
    main()
