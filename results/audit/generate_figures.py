"""Generate audit visualizations for experiment analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set up output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Define experiment data based on audit findings
# Timeline of best results by phase
timeline_data = {
    'Phase': ['Baseline', 'L_trial Best', 'Phase 2\nBest', 'Phase 2c\nBest',
              'Phase 3\nBest', 'Phase 4\nBest', 'Phase 5\nBest', 'Final\nModel'],
    'Val_Corr': [0.2901, 0.3614, 0.3449, 0.3762, 0.3843, 0.4021, 0.4121, 0.4070],
    'Date_Order': [1, 2, 3, 4, 5, 6, 7, 8],
    'Key_Technique': ['None', 'Grad Bal\nScale 0.5', 'Learnable\nAlpha', 'Input\nEmbed 56',
                      'Attention\nEmbed', 'Learnable\nh0', 'Extended\nTraining', '2000\nEpochs']
}

# Figure 1: Fit Quality Over Time
fig1, ax1 = plt.subplots(figsize=(12, 6))

x = timeline_data['Date_Order']
y = timeline_data['Val_Corr']
labels = timeline_data['Phase']
techniques = timeline_data['Key_Technique']

# Plot line
ax1.plot(x, y, 'b-o', linewidth=2, markersize=10)

# Annotate key techniques
for i, (xi, yi, tech) in enumerate(zip(x, y, techniques)):
    if i > 0:  # Skip baseline annotation
        ax1.annotate(tech, (xi, yi), textcoords="offset points",
                    xytext=(0, 15), ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Horizontal line at target
ax1.axhline(y=0.40, color='green', linestyle='--', label='Target (0.40)')

# Labels
ax1.set_xlabel('Experiment Phase', fontsize=12)
ax1.set_ylabel('Validation PSTH Correlation', fontsize=12)
ax1.set_title('Fit Quality Improvement Over Time', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotate improvement
ax1.annotate(f'+{(0.4121/0.2901 - 1)*100:.1f}% total\nimprovement',
             xy=(7, 0.4121), xytext=(6, 0.43),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
fig1.savefig(os.path.join(figures_dir, 'fit_quality_over_time.png'), dpi=150)
plt.close(fig1)
print("Saved: fit_quality_over_time.png")

# Figure 2: Parameter vs Performance (multiple subplots)
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

# 2a: Learning Rate
lr_data = {'LR': [0.0005, 0.001], 'Val_Corr': [0.3694, 0.3762]}
axes[0, 0].bar(range(len(lr_data['LR'])), lr_data['Val_Corr'], color='steelblue')
axes[0, 0].set_xticks(range(len(lr_data['LR'])))
axes[0, 0].set_xticklabels([str(lr) for lr in lr_data['LR']])
axes[0, 0].set_xlabel('Learning Rate')
axes[0, 0].set_ylabel('Val Correlation')
axes[0, 0].set_title('Learning Rate Effect')
axes[0, 0].set_ylim(0.35, 0.40)

# 2b: Embedding Dimension
embed_data = {'Dim': [0, 28, 56, 112, 168],
              'Val_Corr': [0.2901, 0.3341, 0.3762, 0.3440, 0.3120]}
axes[0, 1].plot(embed_data['Dim'], embed_data['Val_Corr'], 'g-o', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Embedding Dimension')
axes[0, 1].set_ylabel('Val Correlation')
axes[0, 1].set_title('Embedding Dimension Effect')
axes[0, 1].axvline(x=56, color='red', linestyle='--', alpha=0.5, label='Optimal')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2c: L_trial Scale
scale_data = {'Scale': [0, 0.25, 0.5, 1.0, 2.0],
              'Val_Corr': [0.3299, 0.3550, 0.3614, 0.3434, 0.2292]}
axes[1, 0].plot(scale_data['Scale'], scale_data['Val_Corr'], 'r-o', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('L_trial Scale')
axes[1, 0].set_ylabel('Val Correlation')
axes[1, 0].set_title('L_trial Scale Effect')
axes[1, 0].axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Optimal')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 2d: Noise Scale
noise_data = {'Noise': [0.05, 0.1, 0.15, 0.3, 0.5],
              'Val_Corr': [0.3377, 0.4021, 0.3501, 0.3850, 0.3840]}
axes[1, 1].plot(noise_data['Noise'], noise_data['Val_Corr'], 'm-o', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Noise Scale')
axes[1, 1].set_ylabel('Val Correlation')
axes[1, 1].set_title('Noise Scale Effect')
axes[1, 1].axvline(x=0.1, color='blue', linestyle='--', alpha=0.5, label='Optimal')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(figures_dir, 'parameter_vs_performance.png'), dpi=150)
plt.close(fig2)
print("Saved: parameter_vs_performance.png")

# Figure 3: Configuration Heatmap
fig3, ax3 = plt.subplots(figsize=(14, 10))

# Define configuration matrix
techniques = ['Baseline', 'Grad Bal', 'L_trial 0.5', 'Embed 56', 'Attention',
              'Learn h0', 'Learn Alpha', 'Low Rank', 'AdamW', 'Extended']

# Which combinations were tested (1=tested, 0=not tested)
# Rows = base config, Cols = additional feature
tested_matrix = np.array([
    # Base  GB   Lt   E56  Att  h0   Alpha LR   AdW  Ext
    [1,    1,    1,    1,    1,    1,    1,    1,    1,    1],   # Baseline (as base)
    [1,    1,    1,    1,    1,    1,    0,    0,    0,    0],   # Grad Bal
    [1,    1,    1,    1,    1,    1,    0,    0,    0,    0],   # L_trial 0.5
    [1,    1,    1,    1,    1,    0,    1,    0,    1,    0],   # Embed 56
    [1,    1,    1,    1,    1,    1,    1,    0,    1,    1],   # Attention
    [1,    1,    1,    1,    1,    1,    0,    1,    1,    1],   # Learn h0
    [1,    1,    1,    0,    0,    0,    1,    1,    0,    0],   # Learn Alpha
    [1,    1,    1,    0,    0,    1,    1,    1,    0,    0],   # Low Rank
    [1,    1,    1,    1,    1,    1,    0,    0,    1,    0],   # AdamW
    [1,    1,    1,    1,    1,    1,    0,    0,    0,    1],   # Extended
])

# Create heatmap
im = ax3.imshow(tested_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Labels
ax3.set_xticks(range(len(techniques)))
ax3.set_yticks(range(len(techniques)))
ax3.set_xticklabels(techniques, rotation=45, ha='right')
ax3.set_yticklabels(techniques)
ax3.set_xlabel('Additional Feature', fontsize=12)
ax3.set_ylabel('Base Configuration', fontsize=12)
ax3.set_title('Configuration Combinations Tested\n(Green=Tested, Red=Not Tested)', fontsize=14)

# Add text annotations
for i in range(len(techniques)):
    for j in range(len(techniques)):
        text = 'Y' if tested_matrix[i, j] else 'N'
        color = 'black' if tested_matrix[i, j] else 'white'
        ax3.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

plt.colorbar(im, ax=ax3, label='Tested (1) / Not Tested (0)')
plt.tight_layout()
fig3.savefig(os.path.join(figures_dir, 'configuration_heatmap.png'), dpi=150)
plt.close(fig3)
print("Saved: configuration_heatmap.png")

# Figure 4: Summary comparison bar chart
fig4, ax4 = plt.subplots(figsize=(14, 8))

# Top experiments
experiments = [
    'extended_training', 'final_model', 'attention_h0',
    'low_rank_50_p5', 'attention_adamw', 'attention_baseline',
    'embed_56_learn', 'gradbal_0.5', 'alpha_neuron', 'baseline'
]
val_corrs = [0.4121, 0.4070, 0.4021, 0.3990, 0.3878, 0.3907, 0.3762, 0.3614, 0.3449, 0.2901]
colors = ['gold', 'gold', 'green', 'steelblue', 'steelblue', 'steelblue',
          'steelblue', 'steelblue', 'steelblue', 'gray']

bars = ax4.barh(range(len(experiments)), val_corrs, color=colors)

# Target line
ax4.axvline(x=0.40, color='red', linestyle='--', linewidth=2, label='Target (0.40)')

# Labels
ax4.set_yticks(range(len(experiments)))
ax4.set_yticklabels(experiments)
ax4.set_xlabel('Validation PSTH Correlation', fontsize=12)
ax4.set_title('Top 10 Experiments by Performance', fontsize=14)
ax4.legend()

# Value labels
for i, (bar, val) in enumerate(zip(bars, val_corrs)):
    ax4.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)

ax4.set_xlim(0.25, 0.45)
plt.tight_layout()
fig4.savefig(os.path.join(figures_dir, 'top_experiments_comparison.png'), dpi=150)
plt.close(fig4)
print("Saved: top_experiments_comparison.png")

print(f"\nAll figures saved to: {figures_dir}")
