#!/usr/bin/env python3
"""
Simplified I→E Connectivity Analysis for Condition-Specific Loss Model

Analyzes whether the conditioned-loss model shows different connectivity
patterns compared to the original model.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import h5py

# Paths
BASE_DIR = Path("/Users/jph/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/4factors-rnn-analysis")
CONDITIONED_WEIGHTS_DIR = BASE_DIR / "results/conditioned_loss_08_15/weights"
ORIGINAL_WEIGHTS_DIR = BASE_DIR / "results/final_model/weights"
DATA_FILE = BASE_DIR / "data/rnn_export_Newton_08_15_2025_SC.mat"
OUTPUT_DIR = BASE_DIR / "results/conditioned_loss_08_15/connectivity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Network structure
N_E = 52  # Excitatory neurons (41 recorded + 11 hidden)
N_I = 13  # Inhibitory neurons (all recorded)
N_RECORDED_E = 41
N_RECORDED = 54


def load_weights(weights_dir):
    """Load weight matrices."""
    W_rec = np.load(weights_dir / "W_rec.npy")
    W_in = np.load(weights_dir / "W_in.npy")
    h0 = np.load(weights_dir / "h0.npy")
    return W_rec, W_in, h0


def compute_gini(x):
    """Compute Gini coefficient."""
    x = np.abs(x)
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def compute_entropy(x):
    """Compute normalized entropy."""
    x = np.abs(x)
    if x.sum() == 0:
        return 0
    p = x / x.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(x))
    return entropy / max_entropy if max_entropy > 0 else 0


def analyze_ie_weights(W_rec, name="Model"):
    """Analyze I→E connectivity structure."""
    # Extract I→E submatrix
    W_IE = W_rec[:N_E, N_E:N_E + N_I]  # (52 E targets, 13 I sources)

    print(f"\n=== {name} I→E Connectivity ===")
    print(f"  Weight range: [{W_IE.min():.4f}, {W_IE.max():.4f}]")
    print(f"  Mean weight: {W_IE.mean():.4f}")
    print(f"  Std weight: {W_IE.std():.4f}")

    # Per-I neuron statistics
    gini_values = []
    entropy_values = []
    n_effective = []

    for i in range(N_I):
        weights = np.abs(W_IE[:, i])
        gini_values.append(compute_gini(weights))
        entropy_values.append(compute_entropy(weights))

        p = weights / (weights.sum() + 1e-8)
        n_eff = 1 / (np.sum(p**2) + 1e-8)
        n_effective.append(n_eff)

    gini_values = np.array(gini_values)
    entropy_values = np.array(entropy_values)
    n_effective = np.array(n_effective)

    print(f"  Gini: mean={gini_values.mean():.4f}, std={gini_values.std():.4f}")
    print(f"  Entropy: mean={entropy_values.mean():.4f}, std={entropy_values.std():.4f}")
    print(f"  N_effective: mean={n_effective.mean():.1f}, std={n_effective.std():.1f}")

    return {
        'W_IE': W_IE,
        'gini': gini_values,
        'entropy': entropy_values,
        'n_effective': n_effective
    }


def compare_models(orig_results, cond_results):
    """Compare connectivity between original and conditioned models."""
    print("\n=== Model Comparison ===")

    # Gini comparison
    t, p = stats.ttest_ind(orig_results['gini'], cond_results['gini'])
    print(f"  Gini: orig={orig_results['gini'].mean():.4f}, cond={cond_results['gini'].mean():.4f}, p={p:.4f}")

    # Entropy comparison
    t, p = stats.ttest_ind(orig_results['entropy'], cond_results['entropy'])
    print(f"  Entropy: orig={orig_results['entropy'].mean():.4f}, cond={cond_results['entropy'].mean():.4f}, p={p:.4f}")

    # N_effective comparison
    t, p = stats.ttest_ind(orig_results['n_effective'], cond_results['n_effective'])
    print(f"  N_effective: orig={orig_results['n_effective'].mean():.1f}, cond={cond_results['n_effective'].mean():.1f}, p={p:.4f}")

    # Weight correlation between models
    r, p = stats.pearsonr(orig_results['W_IE'].flatten(), cond_results['W_IE'].flatten())
    print(f"  Weight correlation between models: r={r:.4f}, p={p:.10f}")


def plot_ie_comparison(orig_results, cond_results, save_path):
    """Plot I→E weight comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Weight matrices
    W_orig = orig_results['W_IE'][:N_RECORDED_E, :]
    W_cond = cond_results['W_IE'][:N_RECORDED_E, :]

    vmax = max(np.abs(W_orig).max(), np.abs(W_cond).max())

    ax = axes[0, 0]
    im = ax.imshow(W_orig, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron')
    ax.set_title('Original Model: I→E Weights')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(W_cond, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron')
    ax.set_title('Conditioned Model: I→E Weights')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    diff = W_cond - W_orig
    vmax_diff = np.abs(diff).max()
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron')
    ax.set_title('Difference (Conditioned - Original)')
    plt.colorbar(im, ax=ax)

    # Bottom row: Statistics
    ax = axes[1, 0]
    x = np.arange(N_I)
    width = 0.35
    ax.bar(x - width/2, orig_results['gini'], width, label='Original', alpha=0.8)
    ax.bar(x + width/2, cond_results['gini'], width, label='Conditioned', alpha=0.8)
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Inhibition Specificity (Gini)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x - width/2, orig_results['n_effective'], width, label='Original', alpha=0.8)
    ax.bar(x + width/2, cond_results['n_effective'], width, label='Conditioned', alpha=0.8)
    ax.axhline(N_E, color='gray', linestyle='--', alpha=0.5, label='Max')
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('Effective Targets')
    ax.set_title('Effective Number of E Targets')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(orig_results['W_IE'].flatten(), cond_results['W_IE'].flatten(), alpha=0.3, s=10)
    ax.plot([-vmax, vmax], [-vmax, vmax], 'k--', alpha=0.5)
    r, _ = stats.pearsonr(orig_results['W_IE'].flatten(), cond_results['W_IE'].flatten())
    ax.set_xlabel('Original Weight')
    ax.set_ylabel('Conditioned Weight')
    ax.set_title(f'Weight Correlation: r={r:.3f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("CONNECTIVITY ANALYSIS - Conditioned vs Original Model")
    print("=" * 60)

    # Load weights
    print("\nLoading original model weights...")
    W_rec_orig, W_in_orig, h0_orig = load_weights(ORIGINAL_WEIGHTS_DIR)

    print("Loading conditioned model weights...")
    W_rec_cond, W_in_cond, h0_cond = load_weights(CONDITIONED_WEIGHTS_DIR)

    # Analyze I→E connectivity
    orig_results = analyze_ie_weights(W_rec_orig, "Original")
    cond_results = analyze_ie_weights(W_rec_cond, "Conditioned")

    # Compare models
    compare_models(orig_results, cond_results)

    # Generate figures
    print("\n=== Generating Figures ===")
    plot_ie_comparison(orig_results, cond_results, OUTPUT_DIR / 'ie_weight_comparison.png')

    # Interpret results
    print("\n=== Interpretation ===")
    gini_change = cond_results['gini'].mean() - orig_results['gini'].mean()
    neff_change = cond_results['n_effective'].mean() - orig_results['n_effective'].mean()

    if gini_change > 0.05:
        print("  Conditioned model has MORE SPECIFIC inhibition (higher Gini)")
    elif gini_change < -0.05:
        print("  Conditioned model has MORE DISTRIBUTED inhibition (lower Gini)")
    else:
        print("  Inhibition specificity similar between models")

    if neff_change > 2:
        print("  Conditioned model inhibits MORE E neurons effectively")
    elif neff_change < -2:
        print("  Conditioned model inhibits FEWER E neurons effectively")
    else:
        print("  Effective target count similar between models")

    print(f"\n  Results saved to: {OUTPUT_DIR}")

    return orig_results, cond_results


if __name__ == '__main__':
    orig_results, cond_results = main()
