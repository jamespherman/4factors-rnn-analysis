"""
Diagnostic script for L_trial loss behavior.

Analyzes:
1. Pairwise distance matrix structure at different training stages
2. Soft-matching assignment degeneracy
3. Comparison of training with and without L_trial
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import EIRNNLoss, compute_L_trial, smooth_temporal
from src.data_loader import load_session, train_val_split


def analyze_ltrial_components(
    model: EIRNN,
    data: dict,
    device: str,
    epoch: int
) -> dict:
    """
    Detailed analysis of L_trial loss components.

    Returns breakdown of:
    - Pairwise distance matrix statistics
    - Soft assignment entropy (degeneracy measure)
    - Effective number of matches
    """
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        mask = data['mask'].to(device)

        model_rates, _ = model(inputs)

        # Get recorded neurons only
        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]

        # Population-average activity per trial
        model_pop = model_rates.mean(dim=2)   # [batch, time]
        target_pop = targets.mean(dim=2)

        # Apply mask
        model_pop = model_pop * mask
        target_pop = target_pop * mask

        # Temporal smoothing (same as in compute_L_trial)
        bin_size_ms = 25.0
        smooth_ms = 32.0
        kernel_size = max(1, int(smooth_ms / bin_size_ms))
        model_pop = smooth_temporal(model_pop, kernel_size, dim=1)
        target_pop = smooth_temporal(target_pop, kernel_size, dim=1)

        # Z-score normalize across trials (per timepoint)
        model_mean = model_pop.mean(dim=0, keepdim=True)
        model_std = model_pop.std(dim=0, keepdim=True) + 1e-8
        model_pop_norm = (model_pop - model_mean) / model_std

        target_mean = target_pop.mean(dim=0, keepdim=True)
        target_std = target_pop.std(dim=0, keepdim=True) + 1e-8
        target_pop_norm = (target_pop - target_mean) / target_std

        # Compute pairwise distances
        distances = torch.cdist(model_pop_norm, target_pop_norm, p=2)  # [batch, batch]

        # Soft assignment weights (same temperature as in loss)
        temperature = 0.1
        soft_weights = F.softmax(-distances / temperature, dim=1)  # [batch, batch]

        # Compute metrics
        n_trials = distances.shape[0]

        # 1. Distance matrix statistics
        dist_mean = distances.mean().item()
        dist_std = distances.std().item()
        dist_min = distances.min().item()
        dist_max = distances.max().item()
        diagonal_mean = distances.diag().mean().item()  # Self-distance (should be low if model matches data)

        # 2. Assignment entropy (per row)
        # High entropy = uniform assignment (degenerate)
        # Low entropy = peaked assignment (good matching)
        row_entropy = -(soft_weights * torch.log(soft_weights + 1e-10)).sum(dim=1)
        mean_entropy = row_entropy.mean().item()
        max_possible_entropy = np.log(n_trials)  # Uniform distribution
        normalized_entropy = mean_entropy / max_possible_entropy

        # 3. Effective number of matches (exp of entropy)
        effective_matches = torch.exp(row_entropy).mean().item()

        # 4. Max weight per row (should be close to 1 for good assignment)
        max_weights = soft_weights.max(dim=1)[0]
        mean_max_weight = max_weights.mean().item()

        # 5. Check for diagonal dominance (model trial i matches data trial i)
        diagonal_weights = soft_weights.diag()
        diagonal_weight_mean = diagonal_weights.mean().item()

        # 6. Actual L_trial value
        matched_distances = (soft_weights * distances).sum(dim=1)
        L_trial = matched_distances.mean().item()

        # 7. Model vs target variance analysis
        model_trial_variance = model_pop_norm.var(dim=0).mean().item()  # Variance across trials per timepoint
        target_trial_variance = target_pop_norm.var(dim=0).mean().item()

        # 8. Check if model trials are collapsing (all similar)
        model_trial_spread = model_pop.std(dim=0).mean().item()  # Raw spread
        target_trial_spread = target_pop.std(dim=0).mean().item()

    return {
        'epoch': epoch,
        'L_trial': L_trial,
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'dist_min': dist_min,
        'dist_max': dist_max,
        'diagonal_dist_mean': diagonal_mean,
        'mean_entropy': mean_entropy,
        'normalized_entropy': normalized_entropy,
        'effective_matches': effective_matches,
        'mean_max_weight': mean_max_weight,
        'diagonal_weight_mean': diagonal_weight_mean,
        'model_trial_variance': model_trial_variance,
        'target_trial_variance': target_trial_variance,
        'model_trial_spread': model_trial_spread,
        'target_trial_spread': target_trial_spread,
        'n_trials': n_trials,
    }


def train_with_diagnostics(
    data_path: str,
    output_dir: str,
    max_epochs: int = 500,
    lambda_trial: float = 0.0,
    device: str = 'cpu',
    seed: int = 42,
    diagnostic_epochs: list = [0, 100, 300]
):
    """
    Train model and collect L_trial diagnostics at specified epochs.
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("="*60)
    print("Loading data...")
    dataset = load_session(data_path)

    # Train/val split
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)
    print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")

    # Get trial batches
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
    print("\n" + "="*60)
    print("Creating model...")
    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()

    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        dt=float(dataset.bin_size_ms),
        device=device
    )

    # Loss function
    loss_fn = EIRNNLoss(
        bin_size_ms=dataset.bin_size_ms,
        lambda_reg=1e-4,
        use_gradient_normalization=True
    )
    loss_fn.lambda_trial = lambda_trial

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    # Training loop
    print("\n" + "="*60)
    print(f"Training with lambda_trial={lambda_trial}...")

    history = {
        'train_total': [], 'val_total': [],
        'train_L_neuron': [], 'val_L_neuron': [],
        'train_L_trial': [], 'val_L_trial': [],
        'train_psth_corr': [], 'val_psth_corr': [],
    }

    diagnostics = []

    best_val_corr = float('-inf')

    pbar = tqdm(range(max_epochs), desc="Training")
    for epoch in pbar:
        # Collect diagnostics at specified epochs
        if epoch in diagnostic_epochs:
            print(f"\n\nCollecting diagnostics at epoch {epoch}...")
            diag = analyze_ltrial_components(model, train_data, device, epoch)
            diagnostics.append(diag)

            print(f"  L_trial = {diag['L_trial']:.4f}")
            print(f"  Distance matrix: mean={diag['dist_mean']:.3f}, std={diag['dist_std']:.3f}")
            print(f"  Diagonal distance (self-match): {diag['diagonal_dist_mean']:.3f}")
            print(f"  Normalized entropy: {diag['normalized_entropy']:.3f} (1.0 = uniform/degenerate)")
            print(f"  Effective matches: {diag['effective_matches']:.1f} / {diag['n_trials']}")
            print(f"  Mean max weight: {diag['mean_max_weight']:.3f} (should be ~1 for good matching)")
            print(f"  Model trial spread: {diag['model_trial_spread']:.3f}")
            print(f"  Target trial spread: {diag['target_trial_spread']:.3f}")
            print()

        # Train
        model.train()
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        optimizer.zero_grad()
        model_rates, model_outputs = model(inputs)

        n_recorded = targets.shape[2]
        loss, components = loss_fn(model, model_rates[:, :, :n_recorded], targets, mask=mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_inputs = val_data['inputs'].to(device)
            val_targets = val_data['targets'].to(device)
            val_mask = val_data['mask'].to(device)

            val_rates, _ = model(val_inputs)
            _, val_components = loss_fn(model, val_rates[:, :, :n_recorded], val_targets, mask=val_mask)

            # PSTH correlation
            model_psth = model_rates[:, :, :n_recorded].mean(dim=0).cpu().numpy()
            target_psth = targets.mean(dim=0).cpu().numpy()

            correlations = []
            for i in range(n_recorded):
                r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
                if not np.isnan(r):
                    correlations.append(r)
            train_psth_corr = np.mean(correlations) if correlations else 0.0

            val_model_psth = val_rates[:, :, :n_recorded].mean(dim=0).cpu().numpy()
            val_target_psth = val_targets.mean(dim=0).cpu().numpy()

            val_correlations = []
            for i in range(n_recorded):
                r = np.corrcoef(val_model_psth[:, i], val_target_psth[:, i])[0, 1]
                if not np.isnan(r):
                    val_correlations.append(r)
            val_psth_corr = np.mean(val_correlations) if val_correlations else 0.0

        scheduler.step(val_components['total'])

        # Record history
        history['train_total'].append(components['total'])
        history['val_total'].append(val_components['total'])
        history['train_L_neuron'].append(components['L_neuron'])
        history['val_L_neuron'].append(val_components['L_neuron'])
        history['train_L_trial'].append(components['L_trial'])
        history['val_L_trial'].append(val_components['L_trial'])
        history['train_psth_corr'].append(train_psth_corr)
        history['val_psth_corr'].append(val_psth_corr)

        # Update progress bar
        pbar.set_postfix({
            'L_n': f"{components['L_neuron']:.3f}",
            'L_t': f"{components['L_trial']:.3f}",
            'corr': f"{val_psth_corr:.3f}",
        })

        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_psth_corr': val_psth_corr,
            }, str(output_dir / 'model_best.pt'))

    # Final diagnostics
    print("\n\nFinal diagnostics at epoch", epoch)
    final_diag = analyze_ltrial_components(model, train_data, device, epoch)
    diagnostics.append(final_diag)

    print(f"  L_trial = {final_diag['L_trial']:.4f}")
    print(f"  Distance matrix: mean={final_diag['dist_mean']:.3f}, std={final_diag['dist_std']:.3f}")
    print(f"  Normalized entropy: {final_diag['normalized_entropy']:.3f}")
    print(f"  Effective matches: {final_diag['effective_matches']:.1f} / {final_diag['n_trials']}")

    # Save results
    results = {
        'lambda_trial': lambda_trial,
        'max_epochs': max_epochs,
        'best_val_corr': best_val_corr,
        'final_val_corr': val_psth_corr,
        'history': history,
        'diagnostics': diagnostics,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Best validation correlation: {best_val_corr:.4f}")
    print(f"Results saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Diagnose L_trial loss behavior')
    parser.add_argument('--data', type=str, required=True, help='Path to exported .mat file')
    parser.add_argument('--output', type=str, default='results/ltrial_diagnostic', help='Output directory')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs')
    parser.add_argument('--lambda-trial', type=float, default=0.0, help='Weight for L_trial (0 = disabled)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    train_with_diagnostics(
        data_path=args.data,
        output_dir=args.output,
        max_epochs=args.epochs,
        lambda_trial=args.lambda_trial,
        device=args.device,
        seed=args.seed,
        diagnostic_epochs=[0, 100, 300]
    )


if __name__ == "__main__":
    main()
