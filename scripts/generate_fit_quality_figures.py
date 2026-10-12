#!/usr/bin/env python3
"""
Generate figures and organize outputs for fit quality improvement experiments.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results' / 'fit_quality_improvement'
FIGURES_DIR = RESULTS_DIR / 'figures'
BEST_MODEL_DIR = RESULTS_DIR / 'best_model'

# Create directories
FIGURES_DIR.mkdir(exist_ok=True)
BEST_MODEL_DIR.mkdir(exist_ok=True)
(BEST_MODEL_DIR / 'weights').mkdir(exist_ok=True)

# Load all results
with open(RESULTS_DIR / 'all_results.json') as f:
    all_results = json.load(f)

# Original results for comparison
ORIGINAL_RESULTS = {
    'E0': 0.1646,
    'E1': 0.1763,
    'E2': 0.1585
}

# Extract improved results
improved_results = {
    'E0': all_results['E0_seed0']['best_val_correlation'],
    'E1': all_results['E1_seed0']['best_val_correlation'],
    'E2': all_results['E2_seed0']['best_val_correlation']
}

print("Results Summary:")
print(f"  E0: {ORIGINAL_RESULTS['E0']:.4f} -> {improved_results['E0']:.4f} (+{100*(improved_results['E0']/ORIGINAL_RESULTS['E0']-1):.1f}%)")
print(f"  E1: {ORIGINAL_RESULTS['E1']:.4f} -> {improved_results['E1']:.4f} (+{100*(improved_results['E1']/ORIGINAL_RESULTS['E1']-1):.1f}%)")
print(f"  E2: {ORIGINAL_RESULTS['E2']:.4f} -> {improved_results['E2']:.4f} (+{100*(improved_results['E2']/ORIGINAL_RESULTS['E2']-1):.1f}%)")

# =============================================================================
# Figure 1: Fit Quality Comparison
# =============================================================================
print("\nGenerating fit_quality_comparison.png...")

fig, ax = plt.subplots(figsize=(10, 6))

experiments = ['E0\n(Baseline)', 'E1\n(Temporal Fix)', 'E2\n(Primary Test)']
x = np.arange(len(experiments))
width = 0.35

original = [ORIGINAL_RESULTS['E0'], ORIGINAL_RESULTS['E1'], ORIGINAL_RESULTS['E2']]
improved = [improved_results['E0'], improved_results['E1'], improved_results['E2']]

bars1 = ax.bar(x - width/2, original, width, label='Original (Basic Model)', color='#1f77b4', alpha=0.7)
bars2 = ax.bar(x + width/2, improved, width, label='Improved (Attention + h0)', color='#2ca02c', alpha=0.7)

ax.set_ylabel('Validation PSTH Correlation', fontsize=12)
ax.set_title('Fit Quality: Original vs Improved Model', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, original):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, improved):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)

# Add improvement percentages
for i, (orig, imp) in enumerate(zip(original, improved)):
    pct = 100 * (imp / orig - 1)
    ax.annotate(f'+{pct:.1f}%', xy=(x[i] + width/2, imp + 0.015),
                ha='center', fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fit_quality_comparison.png', dpi=150)
plt.close()

# =============================================================================
# Figure 2: Learning Curves Comparison
# =============================================================================
print("Generating learning_curves_comparison.png...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (exp_name, ax) in enumerate(zip(['E0', 'E1', 'E2'], axes)):
    exp_dir = RESULTS_DIR / f'{exp_name}_seed0'
    with open(exp_dir / 'history.json') as f:
        history = json.load(f)

    epochs = np.arange(1, len(history['val_correlation']) + 1)

    ax.plot(epochs, history['val_correlation'], 'b-', linewidth=1, alpha=0.7)

    # Mark best epoch
    best_idx = np.argmax(history['val_correlation'])
    best_val = history['val_correlation'][best_idx]
    ax.scatter([best_idx + 1], [best_val], color='red', s=100, zorder=5,
               label=f'Best: {best_val:.4f} @ epoch {best_idx + 1}')

    # Add horizontal line at original
    orig_val = ORIGINAL_RESULTS[exp_name]
    ax.axhline(y=orig_val, color='orange', linestyle='--', alpha=0.7,
               label=f'Original: {orig_val:.4f}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Correlation')
    ax.set_title(f'{exp_name}: Training Progress')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'learning_curves_comparison.png', dpi=150)
plt.close()

# =============================================================================
# Figure 3: Emergent Selectivity (E2)
# =============================================================================
print("Generating emergent_selectivity.png...")

with open(RESULTS_DIR / 'E2_seed0' / 'emergent_selectivity.json') as f:
    selectivity = json.load(f)

fig, ax = plt.subplots(figsize=(10, 6))

factors = ['Reward', 'Probability', 'Salience', 'Identity']
fractions = [
    selectivity['reward']['n_significant_emergent'] / max(selectivity['reward']['n_non_receivers'], 1),
    selectivity['probability']['n_significant_emergent'] / max(selectivity['probability']['n_non_receivers'], 1),
    selectivity['salience']['n_significant_emergent'] / max(selectivity['salience']['n_non_receivers'], 1),
    selectivity['identity']['n_significant_emergent'] / max(selectivity['identity']['n_non_receivers'], 1)
]

# Original E2 results for comparison
original_fractions = [0.50, 0.84, 0.34, 0.41]

x = np.arange(len(factors))
width = 0.35

bars1 = ax.bar(x - width/2, original_fractions, width, label='Original E2', color='#1f77b4', alpha=0.7)
bars2 = ax.bar(x + width/2, fractions, width, label='Improved E2', color='#2ca02c', alpha=0.7)

ax.set_ylabel('Fraction with Significant Selectivity', fontsize=12)
ax.set_title('Emergent Selectivity: Neurons Showing Selectivity for\nFactors They Do NOT Receive as Input', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(factors)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

# Add value labels
for bar, val in zip(bars1, original_fractions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.0%}',
            ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, fractions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.0%}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'emergent_selectivity.png', dpi=150)
plt.close()

# =============================================================================
# Figure 4: Ablation Results
# =============================================================================
print("Generating ablation_results.png...")

with open(RESULTS_DIR / 'E2_seed0' / 'ablation_results.json') as f:
    ablation = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

factors = ['Reward', 'Probability', 'Salience', 'Identity']
factor_keys = ['reward', 'probability', 'salience', 'identity']

# Full recurrence ablation
ax = axes[0]
original_sel = [ablation[k]['full_recurrence']['original_mean_abs_emergent'] for k in factor_keys]
ablated_sel = [ablation[k]['full_recurrence']['ablated_mean_abs_emergent'] for k in factor_keys]

x = np.arange(len(factors))
width = 0.35

bars1 = ax.bar(x - width/2, original_sel, width, label='Original', color='#1f77b4')
bars2 = ax.bar(x + width/2, ablated_sel, width, label='After Ablation', color='#d62728')

ax.set_ylabel('Mean |Selectivity|')
ax.set_title('Full Recurrence Ablation (W_rec = 0)')
ax.set_xticks(x)
ax.set_xticklabels(factors)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# I->E only ablation
ax = axes[1]
original_sel_ie = [ablation[k]['ie_only']['original_mean_abs_emergent'] for k in factor_keys]
ablated_sel_ie = [ablation[k]['ie_only']['ablated_mean_abs_emergent'] for k in factor_keys]

bars1 = ax.bar(x - width/2, original_sel_ie, width, label='Original', color='#1f77b4')
bars2 = ax.bar(x + width/2, ablated_sel_ie, width, label='After I->E Ablation', color='#d62728')

ax.set_ylabel('Mean |Selectivity|')
ax.set_title('I->E Only Ablation')
ax.set_xticks(x)
ax.set_xticklabels(factors)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ablation_results.png', dpi=150)
plt.close()

# =============================================================================
# Copy best model to best_model/
# =============================================================================
print("\nCopying best model (E1) to best_model/...")

# E1 had the highest correlation (0.2208)
best_exp = 'E1_seed0'
best_dir = RESULTS_DIR / best_exp

# Copy weights
for weight_file in ['W_rec.npy', 'W_in.npy', 'h0.npy']:
    src = best_dir / weight_file
    if src.exists():
        shutil.copy(src, BEST_MODEL_DIR / 'weights' / weight_file)

# Copy model checkpoint
shutil.copy(best_dir / 'model_best.pt', BEST_MODEL_DIR / 'model_best.pt')

# Copy config (extract from results.json)
with open(best_dir / 'results.json') as f:
    results = json.load(f)
with open(BEST_MODEL_DIR / 'config.json', 'w') as f:
    json.dump(results['config'], f, indent=2)

# Create summary
summary = {
    'experiment': best_exp,
    'best_val_correlation': results['best_val_correlation'],
    'best_epoch': results['best_epoch'],
    'epochs_trained': results['epochs_trained'],
    'training_time_seconds': results['training_time_seconds']
}
with open(BEST_MODEL_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# =============================================================================
# Remove empty experiments/ directory (data is in E0/E1/E2 folders)
# =============================================================================
print("Removing empty experiments/ directory...")
exp_dir = RESULTS_DIR / 'experiments'
if exp_dir.exists() and not any(exp_dir.iterdir()):
    try:
        exp_dir.rmdir()
    except PermissionError:
        print("  (Could not remove - likely locked by OneDrive)")

print("\nDone! Generated files:")
print(f"  - {FIGURES_DIR / 'fit_quality_comparison.png'}")
print(f"  - {FIGURES_DIR / 'learning_curves_comparison.png'}")
print(f"  - {FIGURES_DIR / 'emergent_selectivity.png'}")
print(f"  - {FIGURES_DIR / 'ablation_results.png'}")
print(f"  - {BEST_MODEL_DIR / 'model_best.pt'}")
print(f"  - {BEST_MODEL_DIR / 'config.json'}")
print(f"  - {BEST_MODEL_DIR / 'summary.json'}")
print(f"  - {BEST_MODEL_DIR / 'weights/'}")
