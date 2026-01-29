#!/usr/bin/env python3
"""
Train E-Only RNN Model for R01 Figure Analysis.

This script trains RNNs that only fit E (classic) neurons, leaving I neurons
as unconstrained latent units. This reveals what inhibitory dynamics the
network "needs" to fit the data, for comparison with recorded interneurons.

Key difference from full model:
- L_neuron only computed over E neurons (first n_classic neurons)
- I neurons are still present in the model (Dale's law enforced)
- After training, we extract the learned I neuron PSTHs

Author: Claude Code
Date: 2025-01-25
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
    EIRNNLoss, smooth_temporal
)
from src.data_loader import load_session, train_val_split


def compute_L_neuron_e_only(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    n_e_neurons: int,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: torch.Tensor = None,
    lambda_scale: float = 0.1,
    lambda_var: float = 0.05
) -> torch.Tensor:
    """
    Compute L_neuron only for E neurons.

    Args:
        model_rates: [batch, time, n_neurons] - Full model rates
        target_rates: [batch, time, n_recorded] - Recorded rates (E and I)
        n_e_neurons: Number of E neurons to fit (only first n_e_neurons)
        bin_size_ms: Bin size in ms
        smooth_ms: Smoothing kernel size in ms
        mask: [batch, time] validity mask
        lambda_scale: Weight for scale loss
        lambda_var: Weight for variance loss

    Returns:
        L_neuron for E neurons only
    """
    # Only use first n_e_neurons from both model and target
    model_e = model_rates[:, :, :n_e_neurons]
    target_e = target_rates[:, :, :n_e_neurons]

    return compute_L_neuron(
        model_e, target_e,
        bin_size_ms=bin_size_ms,
        smooth_ms=smooth_ms,
        mask=mask,
        lambda_scale=lambda_scale,
        lambda_var=lambda_var
    )


class EOnlyLoss(nn.Module):
    """
    Loss function that only fits E neurons.
    I neurons are unconstrained latent units.
    """

    def __init__(
        self,
        n_e_neurons: int,
        bin_size_ms: float = 25.0,
        lambda_reg: float = 1e-4,
        lambda_trial: float = 0.5,
        lambda_scale: float = 0.1,
        lambda_var: float = 0.05,
        use_gradient_normalization: bool = True
    ):
        super().__init__()
        self.n_e_neurons = n_e_neurons
        self.bin_size_ms = bin_size_ms
        self.lambda_reg = lambda_reg
        self.lambda_trial = lambda_trial
        self.lambda_scale = lambda_scale
        self.lambda_var = lambda_var
        self.use_gradient_normalization = use_gradient_normalization

        # Running statistics for gradient normalization
        self.register_buffer('loss_ema', torch.ones(3))  # [L_neuron, L_trial, L_reg]
        self.ema_decay = 0.99

    def forward(
        self,
        model: nn.Module,
        model_rates: torch.Tensor,
        target_rates: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Compute total loss (E neurons only for L_neuron).

        Args:
            model: EIRNN model
            model_rates: [batch, time, n_model_neurons]
            target_rates: [batch, time, n_recorded]
            mask: [batch, time] validity mask

        Returns:
            loss: Scalar total loss
            components: Dict with loss components
        """
        # L_neuron only for E neurons
        L_neuron = compute_L_neuron_e_only(
            model_rates, target_rates,
            n_e_neurons=self.n_e_neurons,
            bin_size_ms=self.bin_size_ms,
            smooth_ms=8.0,
            mask=mask,
            lambda_scale=self.lambda_scale,
            lambda_var=self.lambda_var
        )

        # L_trial computed on E neurons only (population trajectory matching)
        model_e = model_rates[:, :, :self.n_e_neurons]
        target_e = target_rates[:, :, :self.n_e_neurons]
        L_trial = compute_L_trial(
            model_e, target_e,
            self.bin_size_ms,
            smooth_ms=32.0,
            mask=mask
        ) * self.lambda_trial

        # L_reg applies to full model
        L_reg = compute_L_reg(model, self.lambda_reg)

        components = {
            'L_neuron': L_neuron.item(),
            'L_trial': L_trial.item(),
            'L_reg': L_reg.item(),
            'lambda_trial': self.lambda_trial,
        }

        # Combine losses
        if self.use_gradient_normalization:
            with torch.no_grad():
                current_losses = torch.tensor([
                    L_neuron.item(), L_trial.item(), L_reg.item()
                ], device=self.loss_ema.device)
                self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * current_losses

            L_neuron_norm = L_neuron / (self.loss_ema[0] + 1e-8)
            L_trial_norm = L_trial / (self.loss_ema[1] + 1e-8)
            L_reg_norm = L_reg / (self.loss_ema[2] + 1e-8)

            loss = L_neuron_norm + L_trial_norm + L_reg_norm
        else:
            loss = L_neuron + L_trial + L_reg

        components['total'] = loss.item()

        return loss, components


def train_epoch(model, data, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()

    inputs = data['inputs'].to(device)
    targets = data['targets'].to(device)
    mask = data['mask'].to(device)

    optimizer.zero_grad()

    model_rates, model_outputs = model(inputs)

    loss, components = loss_fn(model, model_rates, targets, mask=mask)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return components


def validate(model, data, loss_fn, device):
    """Validate on held-out trials."""
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        mask = data['mask'].to(device)

        model_rates, model_outputs = model(inputs)
        loss, components = loss_fn(model, model_rates, targets, mask=mask)

    return components


def compute_psth_correlation(model, data, device, n_e_only=None):
    """Compute mean PSTH correlation across neurons."""
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        # If n_e_only specified, only compute for E neurons
        if n_e_only is not None:
            n_neurons = n_e_only
        else:
            n_neurons = target_psth.shape[1]

        correlations = []
        for i in range(n_neurons):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

        return np.mean(correlations) if correlations else 0.0


def extract_model_i_psth(model, data, device, n_e_neurons, n_i_neurons):
    """
    Extract the learned I neuron PSTHs from the E-only model.

    The model has n_e + n_i neurons total. The first n_e are fit to data,
    the rest are unconstrained I neurons whose activity we want to analyze.

    Returns:
        model_i_psth: [time, n_i] - Trial-averaged I neuron PSTHs
        model_i_rates: [trials, time, n_i] - Per-trial I neuron rates
    """
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        model_rates, _ = model(inputs)

        # Extract I neuron rates (indices n_e to n_e + n_i)
        model_i_rates = model_rates[:, :, n_e_neurons:n_e_neurons + n_i_neurons].cpu().numpy()

        # Trial-average
        model_i_psth = model_i_rates.mean(axis=0)

    return model_i_psth, model_i_rates


def plot_training_curves(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(history['train_total'], label='Train')
    ax.plot(history['val_total'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.set_title('Total Loss')

    ax = axes[0, 1]
    ax.plot(history['train_L_neuron'], label='Train')
    ax.plot(history['val_L_neuron'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_neuron (E only)')
    ax.legend()
    ax.set_title('PSTH Loss (E Neurons Only)')

    ax = axes[1, 0]
    ax.plot(history['train_L_trial'], label='Train')
    ax.plot(history['val_L_trial'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L_trial')
    ax.legend()
    ax.set_title('Trial-Matching Loss')

    ax = axes[1, 1]
    ax.plot(history['train_psth_corr'], label='Train')
    ax.plot(history['val_psth_corr'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean PSTH Correlation (E only)')
    ax.legend()
    ax.set_title('PSTH Correlation (E Neurons)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_i_neurons(model_i_psth, save_path, bin_size_ms=25.0):
    """Plot the learned I neuron PSTHs."""
    n_time, n_i = model_i_psth.shape
    time_axis = np.arange(n_time) * bin_size_ms / 1000  # seconds

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(min(n_i, 16)):
        ax = axes[i]
        ax.plot(time_axis, model_i_psth[:, i], 'purple', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FR (sp/s)')
        ax.set_title(f'Model I Neuron {i}')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_i, 16):
        axes[i].set_visible(False)

    plt.suptitle('Learned I Neuron PSTHs (Unconstrained in E-Only Model)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(model, optimizer, epoch, metrics, config, path):
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
        }
    }, path)


def train_e_only_model(
    data_path: str,
    output_dir: str,
    max_epochs: int = 1000,
    patience: int = 150,
    lr: float = 1e-3,
    lambda_trial: float = 0.5,
    lambda_reg: float = 1e-4,
    device: str = 'cpu',
    seed: int = 42
):
    """
    Train E-only model.

    Args:
        data_path: Path to session .mat file
        output_dir: Output directory
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        lr: Learning rate
        lambda_trial: Trial matching loss weight
        lambda_reg: Regularization weight
        device: 'cpu' or 'cuda'
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print(f"Loading data from {data_path}...")
    dataset = load_session(data_path)

    # Get neuron info
    neuron_info = dataset.get_neuron_info()
    n_e_recorded = neuron_info['n_exc']  # Number of E neurons in data
    n_i_recorded = neuron_info['n_inh']  # Number of I neurons in data
    n_inputs = dataset.get_input_dim()

    print(f"Data has {n_e_recorded} E neurons and {n_i_recorded} I neurons")

    # Train/val split
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)
    print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")

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

    # Create model - same architecture as full model
    # Model will have n_exc E neurons and n_inh I neurons
    # But we only fit the first n_e_recorded E neurons
    print("\n" + "=" * 60)
    print("Creating E-only model...")

    model = create_model_from_data(
        n_classic=n_e_recorded,
        n_interneuron=n_i_recorded,
        n_inputs=n_inputs,
        enforce_ratio=True,  # Add hidden units for 4:1 ratio
        target_total=200,  # 200 total neurons: 160 E + 40 I
        input_embed_dim=56,
        input_embed_type='attention',
        attention_heads=4,
        learnable_h0=True,
        h0_init=0.1,
        dt=float(dataset.bin_size_ms),
        device=device
    )

    print(f"Model has {model.n_exc} E neurons and {model.n_inh} I neurons")
    print(f"Fitting only first {n_e_recorded} E neurons (recorded)")
    print(f"I neurons ({model.n_inh}) are unconstrained latent units")

    # Loss function - E neurons only
    loss_fn = EOnlyLoss(
        n_e_neurons=n_e_recorded,
        bin_size_ms=dataset.bin_size_ms,
        lambda_reg=lambda_reg,
        lambda_trial=lambda_trial,
        use_gradient_normalization=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    # Config
    config = {
        'data_path': str(data_path),
        'model_type': 'e_only',
        'max_epochs': max_epochs,
        'patience': patience,
        'lr': lr,
        'lambda_trial': lambda_trial,
        'lambda_reg': lambda_reg,
        'seed': seed,
        'n_e_recorded': n_e_recorded,
        'n_i_recorded': n_i_recorded,
        'n_e_model': model.n_exc,
        'n_i_model': model.n_inh,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\n" + "=" * 60)
    print("Training E-only model...")

    history = {
        'train_total': [], 'val_total': [],
        'train_L_neuron': [], 'val_L_neuron': [],
        'train_L_trial': [], 'val_L_trial': [],
        'train_psth_corr': [], 'val_psth_corr': [],
    }

    best_val_corr = float('-inf')
    epochs_without_improvement = 0

    pbar = tqdm(range(max_epochs), desc="Training")
    for epoch in pbar:
        # Train
        train_metrics = train_epoch(model, train_data, loss_fn, optimizer, device)
        train_psth_corr = compute_psth_correlation(model, train_data, device, n_e_only=n_e_recorded)

        # Validate
        val_metrics = validate(model, val_data, loss_fn, device)
        val_psth_corr = compute_psth_correlation(model, val_data, device, n_e_only=n_e_recorded)

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

        # Update progress
        pbar.set_postfix({
            'train': f"{train_metrics['total']:.4f}",
            'val': f"{val_metrics['total']:.4f}",
            'corr': f"{val_psth_corr:.3f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
        })

        # Check improvement
        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics, 'val_psth_corr': val_psth_corr},
                config,
                str(output_dir / 'model_best.pt')
            )
        else:
            epochs_without_improvement += 1

        # Early stopping
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

    # Extract and save model I neuron PSTHs
    print("\nExtracting model I neuron PSTHs...")
    model_i_psth, model_i_rates = extract_model_i_psth(
        model, val_data, device,
        n_e_neurons=model.n_exc,
        n_i_neurons=model.n_inh
    )

    np.save(output_dir / 'model_i_psth.npy', model_i_psth)
    np.save(output_dir / 'model_i_rates.npy', model_i_rates)

    # Also save full model outputs for comparison
    model.eval()
    with torch.no_grad():
        inputs = val_data['inputs'].to(device)
        model_rates, _ = model(inputs)
        model_rates = model_rates.cpu().numpy()

    np.save(output_dir / 'val_model_rates.npy', model_rates)
    np.save(output_dir / 'val_target_rates.npy', val_data['targets'].numpy())

    # Plot model I neurons
    plot_model_i_neurons(model_i_psth, str(output_dir / 'model_i_neurons.png'))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation correlation (E only): {best_val_corr:.4f}")
    print(f"Model I neuron PSTH saved to: {output_dir / 'model_i_psth.npy'}")
    print(f"Outputs saved to: {output_dir}")

    return model, history, model_i_psth


def main():
    parser = argparse.ArgumentParser(description='Train E-only RNN model')
    parser.add_argument('--data', type=str, required=True, help='Path to session .mat file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--patience', type=int, default=150, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda-trial', type=float, default=0.5, help='Trial matching weight')
    parser.add_argument('--reg', type=float, default=1e-4, help='Regularization')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    train_e_only_model(
        data_path=args.data,
        output_dir=args.output,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        lambda_trial=args.lambda_trial,
        lambda_reg=args.reg,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
