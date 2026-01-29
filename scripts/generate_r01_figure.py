#!/usr/bin/env python3
"""
Generate R01 Publication Figure.

Creates a 2x3 panel figure showing:
A: Architecture schematic
B: I→E connectivity vs factor selectivity
C: Model I vs recorded interneuron selectivity
D: Subspace variance decomposition
E: Recorded neuron ROC scatter

Author: Claude Code
Date: 2025-01-25
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
from matplotlib import font_manager
from pathlib import Path
from scipy import stats
import json

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/Users/jph/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/4factors-rnn-analysis")
DATA_DIR = BASE_DIR / 'results/r01_figure/panel_data'
OUTPUT_DIR = BASE_DIR / 'results/r01_figure'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'goal_directed': '#762A83',  # Purple
    'stimulus_driven': '#1B7837',  # Teal/Green
    'neutral': '#969696',  # Gray
    'input_potent': '#2166AC',  # Dark blue
    'e_neurons': '#2166AC',  # Dark blue
    'i_neurons': '#E66101',  # Orange
    'model_i': '#762A83',  # Purple
    'recorded_i': '#969696',  # Gray
}

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# ============================================================================
# Panel A: Architecture Schematic
# ============================================================================

def draw_panel_a(ax):
    """Draw the E-I RNN architecture schematic."""
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Layer positions
    input_x = 1
    recurrent_x = 5
    output_x = 9

    # Draw input layer
    input_y_positions = [2, 4, 6]
    for y in input_y_positions:
        circle = Circle((input_x, y), 0.3, facecolor='lightgray',
                        edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    ax.text(input_x, 0.5, 'Input', ha='center', fontsize=9, fontweight='bold')

    # Draw E neurons (filled blue)
    e_positions = [(recurrent_x - 0.8, 5.5), (recurrent_x + 0.8, 5.5),
                   (recurrent_x - 0.8, 4), (recurrent_x + 0.8, 4)]
    for pos in e_positions:
        circle = Circle(pos, 0.35, facecolor=COLORS['e_neurons'],
                       edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    # Draw I neurons (open with orange border)
    i_positions = [(recurrent_x - 0.5, 2.5), (recurrent_x + 0.5, 2.5)]
    for pos in i_positions:
        circle = Circle(pos, 0.35, facecolor='white',
                       edgecolor=COLORS['i_neurons'], linewidth=2)
        ax.add_patch(circle)

    # Labels for E and I
    ax.text(recurrent_x, 6.5, 'E', ha='center', fontsize=10, fontweight='bold',
            color=COLORS['e_neurons'])
    ax.text(recurrent_x, 1.5, 'I', ha='center', fontsize=10, fontweight='bold',
            color=COLORS['i_neurons'])

    ax.text(recurrent_x, 0.5, 'Recurrent', ha='center', fontsize=9, fontweight='bold')

    # Draw output layer
    output_y_positions = [4, 5]
    for y in output_y_positions:
        circle = Circle((output_x, y), 0.3, facecolor='lightgray',
                        edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    ax.text(output_x, 0.5, 'Output', ha='center', fontsize=9, fontweight='bold')

    # Draw connections

    # Input to recurrent (E and I)
    for iy in input_y_positions:
        for ex, ey in e_positions[:2]:
            ax.annotate('', xy=(ex - 0.35, ey), xytext=(input_x + 0.3, iy),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    for iy in input_y_positions[:2]:
        for ix, iy_pos in i_positions:
            ax.annotate('', xy=(ix - 0.35, iy_pos), xytext=(input_x + 0.3, iy),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # E→E recurrent (blue arrows within E population)
    ax.annotate('', xy=(e_positions[1][0], e_positions[1][1]),
               xytext=(e_positions[0][0] + 0.35, e_positions[0][1]),
               arrowprops=dict(arrowstyle='->', color=COLORS['e_neurons'], lw=1.2))

    # E→I (blue to orange)
    ax.annotate('', xy=(i_positions[0][0], i_positions[0][1] + 0.35),
               xytext=(e_positions[2][0], e_positions[2][1] - 0.35),
               arrowprops=dict(arrowstyle='->', color=COLORS['e_neurons'], lw=1.2))

    # I→E (orange, inhibitory - use flat arrowhead)
    ax.annotate('', xy=(e_positions[3][0], e_positions[3][1] - 0.35),
               xytext=(i_positions[1][0], i_positions[1][1] + 0.35),
               arrowprops=dict(arrowstyle='-|>', color=COLORS['i_neurons'], lw=1.5))

    # I→I
    ax.annotate('', xy=(i_positions[1][0] - 0.3, i_positions[1][1] - 0.15),
               xytext=(i_positions[0][0] + 0.3, i_positions[0][1] - 0.15),
               arrowprops=dict(arrowstyle='-|>', color=COLORS['i_neurons'], lw=1.2,
                              connectionstyle='arc3,rad=0.3'))

    # E to output
    for ex, ey in e_positions[:2]:
        for oy in output_y_positions:
            ax.annotate('', xy=(output_x - 0.3, oy),
                       xytext=(ex + 0.35, ey),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Dale's law annotation
    ax.text(recurrent_x, 7.3, "Dale's Law", ha='center', fontsize=8,
            style='italic', color='gray')

    # Legend
    e_patch = mlines.Line2D([], [], marker='o', color='w',
                           markerfacecolor=COLORS['e_neurons'],
                           markersize=8, label='E neurons')
    i_patch = mlines.Line2D([], [], marker='o', color='w',
                           markerfacecolor='white',
                           markeredgecolor=COLORS['i_neurons'],
                           markeredgewidth=2, markersize=8, label='I neurons')
    ax.legend(handles=[e_patch, i_patch], loc='upper left', fontsize=7,
             frameon=False, bbox_to_anchor=(-0.1, 1.05))


# ============================================================================
# Panel B: I→E Connectivity vs Factor Selectivity
# ============================================================================

def draw_panel_b(ax):
    """Draw I→E connectivity vs factor selectivity scatter."""
    # Load data
    data_file = DATA_DIR / 'panel_b_data.npz'
    stats_file = DATA_DIR / 'panel_b_stats.json'

    if not data_file.exists():
        ax.text(0.5, 0.5, 'Data not available\nRun compute_r01_analyses.py',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('E neuron factor selectivity')
        ax.set_ylabel('Total inhibitory input (a.u.)')
        return

    data = np.load(data_file)
    with open(stats_file) as f:
        stats_data = json.load(f)

    # Use reward selectivity (or combine with salience)
    x = data['e_selectivity_reward']
    y = data['total_i_input']

    # Remove NaN
    valid = ~np.isnan(x)
    x = x[valid]
    y = y[valid]

    # Plot scatter
    ax.scatter(x, y, c=COLORS['neutral'], s=15, alpha=0.5, edgecolors='none')

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='gray', linewidth=1.5, linestyle='-')

    # Horizontal reference line at mean
    ax.axhline(y.mean(), color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

    # Statistics annotation
    ax.text(0.95, 0.95, f'r = {stats_data["r_reward"]:.2f}\np = {stats_data["p_reward"]:.2f}',
           ha='right', va='top', transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('E neuron reward selectivity')
    ax.set_ylabel('Total inhibitory input (a.u.)')
    ax.set_title('I→E connectivity', fontsize=10)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# Panel C: Model vs Recorded Interneuron Selectivity
# ============================================================================

def draw_panel_c(ax):
    """Draw model I vs recorded interneuron selectivity distributions."""
    data_file = DATA_DIR / 'panel_c_data.npz'
    stats_file = DATA_DIR / 'panel_c_stats.json'

    if not data_file.exists():
        ax.text(0.5, 0.5, 'Data not available\nTrain E-only models first',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Factor selectivity')
        ax.set_ylabel('Count')
        return

    data = np.load(data_file)
    with open(stats_file) as f:
        stats_data = json.load(f)

    model_sel = data['model_i_selectivity']
    recorded_sel = data['recorded_i_selectivity']

    # Remove NaN
    model_sel = model_sel[~np.isnan(model_sel)]
    recorded_sel = recorded_sel[~np.isnan(recorded_sel)]

    # Check if we have data
    if len(model_sel) == 0 and len(recorded_sel) == 0:
        ax.text(0.5, 0.5, 'E-only models not trained\nTrain E-only models first',
               ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_xlabel('Factor selectivity')
        ax.set_ylabel('Density')
        ax.set_title('Model vs recorded I neurons', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return

    # Determine bins
    all_data = np.concatenate([model_sel, recorded_sel]) if len(model_sel) > 0 and len(recorded_sel) > 0 else np.array([0.1])
    bins = np.linspace(0, max(0.3, np.percentile(all_data, 95) if len(all_data) > 0 else 0.3), 15)

    # Plot histograms
    if len(model_sel) > 0:
        ax.hist(model_sel, bins=bins, alpha=0.6, color=COLORS['model_i'],
               label='Model I units', density=True, edgecolor='none')
    if len(recorded_sel) > 0:
        ax.hist(recorded_sel, bins=bins, alpha=0.6, color=COLORS['recorded_i'],
               label='Recorded interneurons', density=True, edgecolor='none')

    # Vertical line at chance (0)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    # Statistics annotation (only if valid statistics available)
    if not np.isnan(stats_data.get("ks_stat", np.nan)):
        ax.text(0.95, 0.95, f'KS D = {stats_data["ks_stat"]:.2f}\np = {stats_data["ks_pval"]:.3f}',
               ha='right', va='top', transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Factor selectivity (|partial $\eta^2$|)')
    ax.set_ylabel('Density')
    ax.set_title('Model vs recorded I neurons', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# Panel D: Subspace Variance Decomposition
# ============================================================================

def draw_panel_d(ax):
    """Draw input-potent / input-null variance bar chart."""
    data_file = DATA_DIR / 'panel_d_data.npz'
    stats_file = DATA_DIR / 'panel_d_stats.json'

    if not data_file.exists():
        ax.text(0.5, 0.5, 'Data not available\nRun compute_r01_analyses.py',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel('Proportion of variance')
        return

    data = np.load(data_file)
    with open(stats_file) as f:
        stats_data = json.load(f)

    # Bar positions
    x = np.array([0, 1])
    width = 0.35

    # Data
    potent_means = [stats_data['reward_potent_mean'], stats_data['salience_potent_mean']]
    potent_sems = [stats_data['reward_potent_sem'], stats_data['salience_potent_sem']]
    null_means = [stats_data['reward_null_mean'], stats_data['salience_null_mean']]
    null_sems = [stats_data['reward_null_sem'], stats_data['salience_null_sem']]

    # Plot bars
    bars1 = ax.bar(x - width/2, potent_means, width, yerr=potent_sems,
                  label='Input-potent', color=COLORS['input_potent'],
                  capsize=3, error_kw={'linewidth': 1})
    bars2 = ax.bar(x + width/2, null_means, width, yerr=null_sems,
                  label='Input-null', color=COLORS['neutral'],
                  capsize=3, error_kw={'linewidth': 1})

    ax.set_xticks(x)
    ax.set_xticklabels(['Goal-directed\n(reward)', 'Stimulus-driven\n(salience)'])
    ax.set_ylabel('Proportion of factor variance')
    ax.set_title('Subspace decomposition', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# Panel E: Recorded Neuron ROC Scatter
# ============================================================================

def draw_panel_e(ax):
    """Draw recorded neuron ROC selectivity scatter."""
    data_file = DATA_DIR / 'panel_e_data.npz'
    stats_file = DATA_DIR / 'panel_e_stats.json'

    if not data_file.exists():
        ax.text(0.5, 0.5, 'Data not available\nRun compute_r01_analyses.py',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Stimulus-driven selectivity (salience)')
        ax.set_ylabel('Goal-directed selectivity (reward)')
        return

    data = np.load(data_file)
    with open(stats_file) as f:
        stats_data = json.load(f)

    reward_sel = data['reward_selectivity']
    salience_sel = data['salience_selectivity']
    neuron_type = data['neuron_type']

    # Separate E and I
    e_mask = neuron_type == 'E'
    i_mask = neuron_type == 'I'

    # Plot E neurons (filled gray)
    ax.scatter(salience_sel[e_mask], reward_sel[e_mask],
              c=COLORS['neutral'], s=20, alpha=0.5, label='E neurons',
              edgecolors='none')

    # Plot I neurons (open orange)
    ax.scatter(salience_sel[i_mask], reward_sel[i_mask],
              facecolors='none', edgecolors=COLORS['i_neurons'],
              s=40, linewidths=1.5, label='I neurons')

    # Reference lines at selectivity = 0
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

    # Set limits symmetrically
    max_val = max(np.abs(reward_sel).max(), np.abs(salience_sel).max()) * 1.1
    max_val = min(max_val, 1.0)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_xlabel('Stimulus-driven selectivity')
    ax.set_ylabel('Goal-directed selectivity')
    ax.set_title('Recorded neuron selectivity', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    # Quadrant labels (optional, small text)
    ax.text(0.85, 0.85, 'Goal+\nStim+', ha='center', va='center',
           transform=ax.transAxes, fontsize=6, color='gray', alpha=0.7)
    ax.text(0.15, 0.85, 'Goal+\nStim-', ha='center', va='center',
           transform=ax.transAxes, fontsize=6, color='gray', alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# Main Figure Generation
# ============================================================================

def generate_figure():
    """Generate the complete R01 figure."""
    print("Generating R01 figure...")

    # Create figure: 2 rows x 3 columns, ~7" x 5"
    fig = plt.figure(figsize=(7, 5))

    # Define grid: 2 rows, 3 columns
    # Top row: A, B, C
    # Bottom row: D, E, (empty or small legend)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4,
                         left=0.08, right=0.95, top=0.92, bottom=0.1)

    # Panel A: Architecture (top-left)
    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax_a)
    ax_a.text(-0.1, 1.1, 'A', transform=ax_a.transAxes, fontsize=12,
             fontweight='bold', va='top')

    # Panel B: I→E connectivity (top-middle)
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_b(ax_b)
    ax_b.text(-0.15, 1.1, 'B', transform=ax_b.transAxes, fontsize=12,
             fontweight='bold', va='top')

    # Panel C: Model vs recorded I (top-right)
    ax_c = fig.add_subplot(gs[0, 2])
    draw_panel_c(ax_c)
    ax_c.text(-0.15, 1.1, 'C', transform=ax_c.transAxes, fontsize=12,
             fontweight='bold', va='top')

    # Panel D: Subspace decomposition (bottom-left)
    ax_d = fig.add_subplot(gs[1, 0])
    draw_panel_d(ax_d)
    ax_d.text(-0.15, 1.1, 'D', transform=ax_d.transAxes, fontsize=12,
             fontweight='bold', va='top')

    # Panel E: ROC scatter (bottom-middle, spans 2 columns for larger size)
    ax_e = fig.add_subplot(gs[1, 1:])
    draw_panel_e(ax_e)
    ax_e.text(-0.08, 1.1, 'E', transform=ax_e.transAxes, fontsize=12,
             fontweight='bold', va='top')

    # Save figure
    png_path = OUTPUT_DIR / 'figure_rnn_analysis.png'
    pdf_path = OUTPUT_DIR / 'figure_rnn_analysis.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    """Main function."""
    generate_figure()


if __name__ == "__main__":
    main()
