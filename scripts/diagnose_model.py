"""Diagnostic script to understand model behavior before fixes."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.data_loader import load_session, train_val_split


def diagnose(data_path: str):
    """Run diagnostics on model behavior."""

    # Load data
    dataset = load_session(data_path)
    all_data = dataset.get_all_trials()

    # Data statistics
    targets = all_data['targets']
    print("=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"Target shape: {targets.shape}")
    print(f"Target mean: {targets.mean():.2f} sp/s")
    print(f"Target std: {targets.std():.2f} sp/s")
    print(f"Target max: {targets.max():.2f} sp/s")
    print(f"Target min: {targets.min():.2f} sp/s")
    print(f"Target 99th percentile: {np.percentile(targets.numpy(), 99):.2f} sp/s")

    # Check for outliers
    outlier_threshold = 200  # sp/s
    n_outliers = (targets > outlier_threshold).sum().item()
    print(f"Values > {outlier_threshold} sp/s: {n_outliers} ({100*n_outliers/targets.numel():.3f}%)")

    # PSTH statistics (trial-averaged)
    psth = targets.mean(dim=0)  # [time, neurons]
    print(f"\nPSTH temporal std (mean across neurons): {psth.std(dim=0).mean():.3f}")
    print(f"PSTH neuron std (mean across time): {psth.std(dim=1).mean():.3f}")

    # Create model
    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()

    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        device='cpu'
    )

    print("\n" + "=" * 60)
    print("MODEL BEHAVIOR (untrained)")
    print("=" * 60)

    with torch.no_grad():
        inputs = all_data['inputs'][:10]

        # Test with actual inputs
        rates_actual, _ = model(inputs)
        print(f"Rates with actual input: {rates_actual.mean():.3f} ± {rates_actual.std():.3f} sp/s")

        # Test with zero inputs
        zero_inputs = torch.zeros_like(inputs)
        rates_zero, _ = model(zero_inputs)
        print(f"Rates with zero input: {rates_zero.mean():.3f} ± {rates_zero.std():.3f} sp/s")

        # Test with 10x inputs
        rates_10x, _ = model(inputs * 10)
        print(f"Rates with 10x input: {rates_10x.mean():.3f} ± {rates_10x.std():.3f} sp/s")

        # Input modulation ratio
        modulation = (rates_actual.mean() - rates_zero.mean()) / (rates_zero.mean() + 1e-8)
        print(f"\nInput modulation ratio: {modulation:.2%}")

        # Temporal dynamics check
        rates_temporal_std = rates_actual.std(dim=1).mean()  # std across time
        target_temporal_std = targets[:10].std(dim=1).mean()
        print(f"\nModel temporal std: {rates_temporal_std:.3f}")
        print(f"Target temporal std: {target_temporal_std:.3f}")
        print(f"Ratio: {rates_temporal_std/target_temporal_std:.2%}")

    # Weight statistics
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)
    W_rec = model.W_rec.detach().numpy()
    W_in = model.W_in.detach().numpy()

    print(f"W_rec: shape {W_rec.shape}, range [{W_rec.min():.4f}, {W_rec.max():.4f}]")
    print(f"W_in: shape {W_in.shape}, range [{W_in.min():.4f}, {W_in.max():.4f}]")
    print(f"W_rec spectral radius: {np.max(np.abs(np.linalg.eigvals(W_rec))):.4f}")

    # Input signal statistics
    print("\n" + "=" * 60)
    print("INPUT SIGNAL STATISTICS")
    print("=" * 60)
    inputs_all = all_data['inputs']
    print(f"Inputs shape: {inputs_all.shape}")
    print(f"Inputs mean: {inputs_all.mean():.4f}")
    print(f"Inputs std: {inputs_all.std():.4f}")
    print(f"Inputs range: [{inputs_all.min():.4f}, {inputs_all.max():.4f}]")

    # Per-channel statistics
    for i in range(min(5, inputs_all.shape[2])):
        ch = inputs_all[:, :, i]
        print(f"  Channel {i}: mean={ch.mean():.4f}, std={ch.std():.4f}, nonzero={(ch != 0).float().mean():.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .mat file")
    args = parser.parse_args()

    diagnose(args.data)
