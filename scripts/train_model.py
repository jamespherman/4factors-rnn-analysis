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


def train(
    data_path: str,
    output_dir: str,
    max_epochs: int = 1000,
    patience: int = 100,
    lr: float = 1e-3,
    lambda_neuron: float = 1.0,
    lambda_trial: float = 1.0,
    lambda_reg: float = 1e-4,
    normalize_psth: bool = True,
    enforce_ratio: bool = True,
    device: str = 'cpu',
    seed: int = 42
):
    """
    Main training function.

    Args:
        data_path: Path to exported .mat file
        output_dir: Directory for outputs
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        lr: Learning rate
        lambda_neuron: Weight for PSTH (L_neuron) loss
        lambda_trial: Weight for trial-matching (L_trial) loss
        lambda_reg: Regularization strength
        normalize_psth: If True, use z-score normalized PSTH loss (shape only).
                       If False, use raw MSE (scale + shape).
        enforce_ratio: Whether to enforce 4:1 E:I ratio
        device: 'cpu' or 'cuda'
        seed: Random seed
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
        dt=float(dataset.bin_size_ms),
        device=device
    )
    
    # Loss function
    loss_fn = EIRNNLoss(
        bin_size_ms=dataset.bin_size_ms,
        lambda_neuron=lambda_neuron,
        lambda_trial=lambda_trial,
        lambda_reg=lambda_reg,
        normalize_psth=normalize_psth
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
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    pbar = tqdm(range(max_epochs), desc="Training")
    for epoch in pbar:
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
        
        # Update progress bar
        pbar.set_postfix({
            'train': f"{train_metrics['total']:.4f}",
            'val': f"{val_metrics['total']:.4f}",
            'corr': f"{val_psth_corr:.3f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
        })
        
        # Check for improvement
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
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
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
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
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final PSTH correlation: {val_psth_corr:.3f}")
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
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
