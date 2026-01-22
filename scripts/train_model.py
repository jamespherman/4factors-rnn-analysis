"""
Training script for E-I RNN.

Usage:
    python scripts/train_model.py --data path/to/data.mat --output results/

See specs/TRAINING_SPEC.md for details.
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
from src.losses import EIRNNLoss
from src.data_loader import load_session, train_val_split


def train_epoch(
    model: EIRNN,
    data: dict,
    loss_fn: EIRNNLoss,
    optimizer: torch.optim.Optimizer,
    device: str
) -> dict:
    """Train for one epoch (full batch)."""
    model.train()
    
    inputs = data['inputs'].to(device)
    targets = data['targets'].to(device)
    mask = data['mask'].to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    model_rates, model_outputs = model(inputs)
    
    # Compute loss (only for recorded neurons)
    n_recorded = targets.shape[2]
    loss, components = loss_fn(
        model, 
        model_rates[:, :, :n_recorded], 
        targets,
        mask=mask
    )
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Verify constraints still hold
    if not model.verify_constraints():
        print("WARNING: Constraints violated after optimizer step!")
    
    return components


def validate(
    model: EIRNN,
    data: dict,
    loss_fn: EIRNNLoss,
    device: str
) -> dict:
    """Validate on held-out trials."""
    model.eval()
    
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        mask = data['mask'].to(device)
        
        model_rates, model_outputs = model(inputs)
        
        n_recorded = targets.shape[2]
        loss, components = loss_fn(
            model,
            model_rates[:, :, :n_recorded],
            targets,
            mask=mask
        )
    
    return components


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
        model_psth = model_rates.mean(dim=0).cpu().numpy()  # [time, neurons]
        target_psth = targets.mean(dim=0).cpu().numpy()
        
        # Correlation per neuron
        n_recorded = target_psth.shape[1]
        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)
        
        return np.mean(correlations) if correlations else 0.0


def save_checkpoint(
    model: EIRNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    path: str
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'model_config': {
            'n_exc': model.n_exc,
            'n_inh': model.n_inh,
            'n_inputs': model.n_inputs,
            'n_outputs': model.n_outputs,
            'tau': model.tau,
            'dt': model.dt,
            'noise_scale': model.noise_scale,
        }
    }, path)


def plot_training_curves(history: dict, save_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(history['train_total'], label='Train')
    ax.plot(history['val_total'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.set_title('Total Loss')
    
    # L_neuron
    ax = axes[0, 1]
    ax.plot(history['train_L_neuron'], label='Train')
    ax.plot(history['val_L_neuron'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_neuron')
    ax.legend()
    ax.set_title('PSTH Loss')
    
    # L_trial
    ax = axes[1, 0]
    ax.plot(history['train_L_trial'], label='Train')
    ax.plot(history['val_L_trial'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_trial')
    ax.legend()
    ax.set_title('Trial-Matching Loss')
    
    # PSTH correlation
    ax = axes[1, 1]
    ax.plot(history['train_psth_corr'], label='Train')
    ax.plot(history['val_psth_corr'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean PSTH Correlation')
    ax.legend()
    ax.set_title('PSTH Correlation')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_psth_comparison(
    model: EIRNN,
    data: dict,
    neuron_info: dict,
    device: str,
    save_path: str,
    n_best: int = 3,
    n_worst: int = 3
):
    """
    Plot PSTH comparison for best and worst fitting neurons.

    Args:
        model: Trained EIRNN model
        data: Data dict with inputs, targets, mask
        neuron_info: Dict with n_exc, n_inh counts
        device: Device to run model on
        save_path: Path to save the figure
        n_best: Number of best-fitting neurons to plot
        n_worst: Number of worst-fitting neurons to plot
    """
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Trial-average
        model_psth = model_rates.mean(dim=0).cpu().numpy()  # [time, neurons]
        target_psth = targets.mean(dim=0).cpu().numpy()

    # Compute correlation per neuron
    n_recorded = target_psth.shape[1]
    correlations = []
    for i in range(n_recorded):
        r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
        correlations.append(r if not np.isnan(r) else 0.0)
    correlations = np.array(correlations)

    # Get neuron types
    n_exc = neuron_info['n_exc']
    neuron_types = ['E'] * n_exc + ['I'] * (n_recorded - n_exc)

    # Sort by correlation
    sorted_idx = np.argsort(correlations)[::-1]  # Best first

    best_idx = sorted_idx[:n_best]
    worst_idx = sorted_idx[-n_worst:][::-1]  # Worst last

    # Time axis (assuming 25ms bins)
    time_bins = np.arange(target_psth.shape[0]) * 25 / 1000  # Convert to seconds

    # Create figure
    n_rows = n_best + n_worst
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows))

    # Plot best neurons
    for i, idx in enumerate(best_idx):
        ax = axes[i]
        ax.plot(time_bins, target_psth[:, idx], 'b-', linewidth=2, label='Target')
        ax.plot(time_bins, model_psth[:, idx], 'r--', linewidth=2, label='Model')
        ax.set_ylabel('Firing Rate\n(sp/s)')
        ax.set_title(f'BEST #{i+1}: Neuron {idx} ({neuron_types[idx]}) - r = {correlations[idx]:.3f}',
                    fontsize=12, fontweight='bold', color='green')
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Plot worst neurons
    for i, idx in enumerate(worst_idx):
        ax = axes[n_best + i]
        ax.plot(time_bins, target_psth[:, idx], 'b-', linewidth=2, label='Target')
        ax.plot(time_bins, model_psth[:, idx], 'r--', linewidth=2, label='Model')
        ax.set_ylabel('Firing Rate\n(sp/s)')
        ax.set_title(f'WORST #{i+1}: Neuron {idx} ({neuron_types[idx]}) - r = {correlations[idx]:.3f}',
                    fontsize=12, fontweight='bold', color='red')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"PSTH comparison plot saved to: {save_path}")

    # Return summary stats
    return {
        'best_idx': best_idx.tolist(),
        'worst_idx': worst_idx.tolist(),
        'best_corr': correlations[best_idx].tolist(),
        'worst_corr': correlations[worst_idx].tolist(),
        'best_types': [neuron_types[i] for i in best_idx],
        'worst_types': [neuron_types[i] for i in worst_idx],
    }


def train(
    data_path: str,
    output_dir: str,
    max_epochs: int = 1000,
    patience: int = 100,
    lr: float = 1e-3,
    lambda_neuron: float = 1.0,
    lambda_trial: float = 1.0,
    lambda_reg: float = 1e-3,
    normalize_psth: bool = True,
    enforce_ratio: bool = True,
    device: str = 'cpu',
    seed: int = 42,
    warmup_epochs: int = 0,  # Disabled by default - warmup hurts performance
    warmup_lambda_scale: float = 0.3,
    warmup_lambda_var: float = 0.1,
    post_warmup_lambda_scale: float = 0.1,
    post_warmup_lambda_var: float = 0.05,
    trial_ramp_epochs: int = 300,
    bypass_dale: bool = False,
    target_total: int = None
):
    """
    Main training function with curriculum learning.

    Curriculum Learning Strategy:
    - Warmup phase (epochs 0 to warmup_epochs): L_neuron only, higher scale/var weights
    - Ramp phase (warmup_epochs to warmup_epochs + trial_ramp_epochs):
      Gradually ramp L_trial from 0 to lambda_trial, transition lambda weights
    - Full phase: All losses at full weight

    Args:
        data_path: Path to exported .mat file
        output_dir: Directory for outputs
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        lr: Learning rate
        lambda_neuron: Weight for PSTH (L_neuron) loss
        lambda_trial: Weight for trial-matching (L_trial) loss (default 0.1 since raw is ~50x L_neuron)
        lambda_reg: Regularization strength
        normalize_psth: If True, use z-score normalized PSTH loss (shape only).
                       If False, use raw MSE (scale + shape).
        enforce_ratio: Whether to enforce 4:1 E:I ratio
        device: 'cpu' or 'cuda'
        seed: Random seed
        warmup_epochs: Number of epochs for warmup (L_neuron only). Set to 0 to disable warmup.
        warmup_lambda_scale: Lambda for scale loss during warmup
        warmup_lambda_var: Lambda for variance loss during warmup
        post_warmup_lambda_scale: Lambda for scale loss after warmup
        post_warmup_lambda_var: Lambda for variance loss after warmup
        trial_ramp_epochs: Epochs to ramp L_trial from 0 to lambda_trial after warmup (default 300)
        bypass_dale: If True, disable Dale's law constraints (for debugging)
        target_total: If specified, use this many total units (e.g., 200 for 160E+40I with 4:1 ratio)
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
        enforce_ratio=enforce_ratio,
        bypass_dale=bypass_dale,
        target_total=target_total,
        dt=float(dataset.bin_size_ms),
        device=device
    )
    
    # Loss function
    loss_fn = EIRNNLoss(
        bin_size_ms=dataset.bin_size_ms,
        lambda_reg=lambda_reg,
        use_gradient_normalization=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )
    
    # Config for saving
    config = {
        'data_path': str(data_path),
        'max_epochs': max_epochs,
        'patience': patience,
        'lr': lr,
        'lambda_neuron': lambda_neuron,
        'lambda_trial': lambda_trial,
        'lambda_reg': lambda_reg,
        'normalize_psth': normalize_psth,
        'enforce_ratio': enforce_ratio,
        'seed': seed,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'timestamp': datetime.now().isoformat(),
        'warmup_epochs': warmup_epochs,
        'warmup_lambda_scale': warmup_lambda_scale,
        'warmup_lambda_var': warmup_lambda_var,
        'post_warmup_lambda_scale': post_warmup_lambda_scale,
        'post_warmup_lambda_var': post_warmup_lambda_var,
        'trial_ramp_epochs': trial_ramp_epochs,
        'bypass_dale': bypass_dale,
        'target_total': target_total,
    }
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "="*60)
    print("Training...")
    
    history = {
        'train_total': [], 'val_total': [],
        'train_L_neuron': [], 'val_L_neuron': [],
        'train_L_trial': [], 'val_L_trial': [],
        'train_psth_corr': [], 'val_psth_corr': [],
        'stage': [],  # Track curriculum stage
    }

    best_val_corr = float('-inf')  # Track best correlation (maximize)
    epochs_without_improvement = 0

    # Initialize warmup settings
    loss_fn.lambda_trial = 0.0  # No trial loss during warmup
    loss_fn.lambda_scale = warmup_lambda_scale
    loss_fn.lambda_var = warmup_lambda_var

    if warmup_epochs > 0:
        print(f"\nCurriculum learning schedule:")
        print(f"  Warmup (epochs 0-{warmup_epochs}): L_neuron only, lambda_scale={warmup_lambda_scale}, lambda_var={warmup_lambda_var}")
        print(f"  Ramp (epochs {warmup_epochs}-{warmup_epochs + trial_ramp_epochs}): L_trial 0 -> {lambda_trial}")
        print(f"  Full (epochs {warmup_epochs + trial_ramp_epochs}+): All losses at full weight")
    else:
        print(f"\nNo warmup - all losses active from start (lambda_trial={lambda_trial})")

    pbar = tqdm(range(max_epochs), desc="Warmup")
    for epoch in pbar:
        # Curriculum learning: fixed epoch-based transitions
        if epoch < warmup_epochs:
            # Warmup phase: L_neuron only
            stage = "warmup"
            loss_fn.lambda_trial = 0.0
            loss_fn.lambda_scale = warmup_lambda_scale
            loss_fn.lambda_var = warmup_lambda_var
        elif epoch < warmup_epochs + trial_ramp_epochs:
            # Ramp phase: gradually introduce L_trial, transition lambdas
            ramp_progress = (epoch - warmup_epochs) / trial_ramp_epochs
            loss_fn.lambda_trial = lambda_trial * ramp_progress
            # Smoothly transition lambda_scale and lambda_var
            loss_fn.lambda_scale = warmup_lambda_scale + (post_warmup_lambda_scale - warmup_lambda_scale) * ramp_progress
            loss_fn.lambda_var = warmup_lambda_var + (post_warmup_lambda_var - warmup_lambda_var) * ramp_progress
            stage = f"ramp"
        else:
            # Full phase: all losses at target weights
            stage = "full"
            loss_fn.lambda_trial = lambda_trial
            loss_fn.lambda_scale = post_warmup_lambda_scale
            loss_fn.lambda_var = post_warmup_lambda_var

        # Train
        train_metrics = train_epoch(model, train_data, loss_fn, optimizer, device)
        train_psth_corr = compute_psth_correlation(model, train_data, device)

        # Validate
        val_metrics = validate(model, val_data, loss_fn, device)
        val_psth_corr = compute_psth_correlation(model, val_data, device)

        # Update scheduler
        scheduler.step(val_metrics['total'])

        # Record history
        history['train_total'].append(train_metrics['total'])
        history['val_total'].append(val_metrics['total'])
        history['train_L_neuron'].append(train_metrics['L_neuron'])
        history['val_L_neuron'].append(val_metrics['L_neuron'])
        history['train_L_trial'].append(train_metrics['L_trial'])
        history['val_L_trial'].append(val_metrics['L_trial'])
        history['train_psth_corr'].append(train_psth_corr)
        history['val_psth_corr'].append(val_psth_corr)
        history['stage'].append(stage)

        # Update progress bar
        pbar.set_description(stage.capitalize() if stage != "full" else "Training")
        pbar.set_postfix({
            'train': f"{train_metrics['total']:.4f}",
            'val': f"{val_metrics['total']:.4f}",
            'corr': f"{val_psth_corr:.3f}",
            'λ_trial': f"{loss_fn.lambda_trial:.2f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
        })
        
        # Check for improvement (using correlation - maximize)
        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0

            # Save best model
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics, 'val_psth_corr': val_psth_corr},
                config,
                str(output_dir / 'model_best.pt')
            )
        else:
            epochs_without_improvement += 1

        # Early stopping (based on correlation)
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best val_corr={best_val_corr:.4f})")
            break
        
        # Periodic checkpoint
        if (epoch + 1) % 100 == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                config,
                str(output_dir / f'model_epoch{epoch+1}.pt')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch,
        {'train': train_metrics, 'val': val_metrics},
        config,
        str(output_dir / 'model_final.pt')
    )
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    plot_training_curves(history, str(output_dir / 'training_curves.png'))

    # Plot PSTH comparison for best/worst neurons
    print("\nGenerating PSTH comparison plots...")
    psth_summary = plot_psth_comparison(
        model, val_data, neuron_info, device,
        str(output_dir / 'psth_comparison.png'),
        n_best=3, n_worst=3
    )

    # Save PSTH summary
    with open(output_dir / 'psth_summary.json', 'w') as f:
        json.dump(psth_summary, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation correlation: {best_val_corr:.4f}")
    print(f"Final PSTH correlation: {val_psth_corr:.3f}")
    print(f"Final stage: {stage}")
    print(f"Outputs saved to: {output_dir}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train E-I RNN on SC neural data')
    parser.add_argument('--data', type=str, required=True, help='Path to exported .mat file')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-neuron', type=float, default=1.0, help='Weight for PSTH loss')
    parser.add_argument('--lambda-trial', type=float, default=1.0, help='Weight for trial-matching loss')
    parser.add_argument('--reg', type=float, default=1e-4, help='Regularization strength')
    parser.add_argument('--raw-psth', action='store_true', help='Use raw MSE for PSTH loss (learn scale + shape)')
    parser.add_argument('--no-ratio', action='store_true', help='Disable 4:1 E:I ratio enforcement')
    parser.add_argument('--bypass-dale', action='store_true', help='Disable Dale\'s law constraints (for debugging)')
    parser.add_argument('--target-total', type=int, default=None, help='Target total units (e.g., 200 for 160E+40I)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        lambda_neuron=args.lambda_neuron,
        lambda_trial=args.lambda_trial,
        lambda_reg=args.reg,
        normalize_psth=not args.raw_psth,
        enforce_ratio=not args.no_ratio,
        bypass_dale=args.bypass_dale,
        target_total=args.target_total,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
