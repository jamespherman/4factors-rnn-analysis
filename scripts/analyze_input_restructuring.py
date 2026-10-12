#!/usr/bin/env python3
"""
Post-Training Analysis for Input Restructuring Experiments

Generates figures and report after experiments complete.

Usage:
    python scripts/analyze_input_restructuring.py
    python scripts/analyze_input_restructuring.py --results-dir results/input_restructuring

Output:
    results/input_restructuring/analysis/
    results/input_restructuring/REPORT.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input_assignment import InputAssignment, load_assignment


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_results(results_dir: Path) -> Dict[str, Any]:
    """Load results from all experiment runs."""
    results = {}

    # Find all experiment directories
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ['analysis', 'figures']:
            continue

        run_name = exp_dir.name

        # Load results
        result_file = exp_dir / 'results.json'
        if result_file.exists():
            with open(result_file) as f:
                run_results = json.load(f)
        else:
            run_results = {'error': 'No results.json found'}

        # Load selectivity
        sel_file = exp_dir / 'emergent_selectivity.json'
        if sel_file.exists():
            with open(sel_file) as f:
                run_results['selectivity'] = json.load(f)

        # Load ablation
        abl_file = exp_dir / 'ablation_results.json'
        if abl_file.exists():
            with open(abl_file) as f:
                run_results['ablation'] = json.load(f)

        # Load assignment
        assign_file = exp_dir / 'input_assignment.npy'
        if assign_file.exists():
            run_results['assignment'] = load_assignment(str(assign_file))

        # Load weights
        wrec_file = exp_dir / 'W_rec.npy'
        if wrec_file.exists():
            run_results['W_rec'] = np.load(wrec_file)

        results[run_name] = run_results

    return results


def extract_experiment_metrics(results: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Group results by experiment type."""
    by_experiment = {}

    for run_name, run_results in results.items():
        if 'error' in run_results:
            continue

        exp_type = run_results.get('experiment_type', run_name.split('_seed')[0])

        if exp_type not in by_experiment:
            by_experiment[exp_type] = []

        by_experiment[exp_type].append({
            'run_name': run_name,
            'seed': run_results.get('seed', 0),
            'val_corr': run_results.get('best_val_correlation', 0),
            'epochs': run_results.get('epochs_trained', 0),
            'selectivity': run_results.get('selectivity', {}),
            'ablation': run_results.get('ablation', {}),
            'W_rec': run_results.get('W_rec')
        })

    return by_experiment


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_fit_quality_comparison(by_experiment: Dict, save_path: Path):
    """Bar chart comparing validation correlation across experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_order = ['E0', 'E1', 'E2', 'E3a', 'E3b', 'E4']
    exp_names = []
    means = []
    stds = []
    colors = []

    color_map = {
        'E0': '#1f77b4',  # blue
        'E1': '#2ca02c',  # green
        'E2': '#d62728',  # red
        'E3a': '#ff7f0e', # orange
        'E3b': '#9467bd', # purple
        'E4': '#8c564b',  # brown
    }

    for exp_type in exp_order:
        if exp_type not in by_experiment:
            continue

        runs = by_experiment[exp_type]
        corrs = [r['val_corr'] for r in runs]

        exp_names.append(exp_type)
        means.append(np.mean(corrs))
        stds.append(np.std(corrs) if len(corrs) > 1 else 0)
        colors.append(color_map.get(exp_type, 'gray'))

    x = np.arange(len(exp_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    # Add baseline reference line (80% of E0)
    if 'E0' in exp_names:
        e0_idx = exp_names.index('E0')
        threshold = means[e0_idx] * 0.8
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'80% of E0 ({threshold:.3f})')

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names)
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('Fit Quality Across Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample sizes
    for i, (exp, n) in enumerate(zip(exp_names, [len(by_experiment[e]) for e in exp_names])):
        ax.annotate(f'n={n}', (i, means[i] + stds[i] + 0.02),
                   ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_emergent_selectivity_by_factor(by_experiment: Dict, save_path: Path):
    """Bar chart showing fraction of significant emergent selectivity per factor."""
    factors = ['reward', 'probability', 'salience', 'identity']
    exp_order = ['E0', 'E1', 'E2', 'E3a', 'E3b', 'E4']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, factor in zip(axes, factors):
        exp_names = []
        means = []
        stds = []

        for exp_type in exp_order:
            if exp_type not in by_experiment:
                continue

            runs = by_experiment[exp_type]
            fracs = []

            for run in runs:
                sel = run.get('selectivity', {}).get(factor, {})
                frac = sel.get('fraction_significant', 0)
                fracs.append(frac)

            if fracs:
                exp_names.append(exp_type)
                means.append(np.mean(fracs))
                stds.append(np.std(fracs) if len(fracs) > 1 else 0)

        x = np.arange(len(exp_names))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names)
        ax.set_ylabel('Fraction Significant')
        ax.set_title(f'{factor.capitalize()} - Emergent Selectivity')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_ablation_results(by_experiment: Dict, save_path: Path):
    """Bar chart showing selectivity before/after ablation."""
    # Focus on E2 experiments
    if 'E2' not in by_experiment:
        print("  Skipping ablation plot (no E2 results)")
        return

    runs = by_experiment['E2']
    factors = ['reward', 'probability', 'salience', 'identity']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, factor in zip(axes, factors):
        original = []
        ablated_full = []
        ablated_ie = []

        for run in runs:
            abl = run.get('ablation', {}).get(factor, {})
            if not abl:
                continue

            if 'full_recurrence' in abl:
                original.append(abl['full_recurrence']['original_mean_abs_emergent'])
                ablated_full.append(abl['full_recurrence']['ablated_mean_abs_emergent'])

            if 'ie_only' in abl:
                ablated_ie.append(abl['ie_only']['ablated_mean_abs_emergent'])

        if not original:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{factor.capitalize()}')
            continue

        x = np.arange(3)
        width = 0.6
        values = [np.mean(original), np.mean(ablated_full), np.mean(ablated_ie) if ablated_ie else 0]
        errors = [np.std(original) if len(original) > 1 else 0,
                  np.std(ablated_full) if len(ablated_full) > 1 else 0,
                  np.std(ablated_ie) if len(ablated_ie) > 1 else 0]

        bars = ax.bar(x, values, width, yerr=errors, capsize=5,
                     color=['#1f77b4', '#d62728', '#ff7f0e'], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(['Original', 'Full Ablation', 'I->E Only'])
        ax.set_ylabel('Mean |Emergent Selectivity|')
        ax.set_title(f'{factor.capitalize()}')
        ax.grid(True, alpha=0.3, axis='y')

        # Add reduction percentage
        if values[0] > 0:
            reduction = (values[0] - values[1]) / values[0] * 100
            ax.annotate(f'-{reduction:.0f}%', (1, values[1] + errors[1] + 0.01),
                       ha='center', fontsize=10, color='red')

    plt.suptitle('Ablation Analysis (E2 Experiments)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_cross_seed_consistency(by_experiment: Dict, save_path: Path):
    """Correlation matrix of I->E weights across seeds."""
    if 'E2' not in by_experiment or len(by_experiment['E2']) < 2:
        print("  Skipping cross-seed plot (need multiple E2 seeds)")
        return

    runs = by_experiment['E2']
    n_runs = len(runs)

    # Extract I->E weights from each run
    ie_weights = []
    labels = []

    for run in runs:
        W_rec = run.get('W_rec')
        if W_rec is None:
            continue

        # Assume 80% E, 20% I (standard ratio)
        n_total = W_rec.shape[0]
        n_exc = int(0.8 * n_total)

        # I->E weights: rows 0:n_exc, cols n_exc:n_total
        w_ie = W_rec[:n_exc, n_exc:].flatten()
        ie_weights.append(w_ie)
        labels.append(f"seed_{run['seed']}")

    if len(ie_weights) < 2:
        print("  Skipping cross-seed plot (insufficient weight data)")
        return

    # Compute correlation matrix
    n = len(ie_weights)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(ie_weights[i], ie_weights[j])[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add correlation values
    for i in range(n):
        for j in range(n):
            text = f'{corr_matrix[i,j]:.2f}'
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    ax.set_title('I->E Weight Correlation Across E2 Seeds')
    plt.colorbar(im, ax=ax, label='Pearson r')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_ie_weight_structure(by_experiment: Dict, save_path: Path):
    """Heatmap of I->E weights for a representative model."""
    # Use E2_seed0 if available
    if 'E2' not in by_experiment:
        print("  Skipping I->E weight plot (no E2 results)")
        return

    run = by_experiment['E2'][0]  # First seed
    W_rec = run.get('W_rec')

    if W_rec is None:
        print("  Skipping I->E weight plot (no weights)")
        return

    n_total = W_rec.shape[0]
    n_exc = int(0.8 * n_total)
    n_inh = n_total - n_exc

    # Extract submatrices
    W_IE = W_rec[:n_exc, n_exc:]  # I->E (these are negative due to Dale's law)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # I->E heatmap
    ax = axes[0]
    vmax = np.abs(W_IE).max()
    im = ax.imshow(W_IE, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('I neurons (source)')
    ax.set_ylabel('E neurons (target)')
    ax.set_title(f'I->E Weights (E2_seed{run["seed"]})')
    plt.colorbar(im, ax=ax, label='Weight')

    # Distribution
    ax = axes[1]
    ax.hist(W_IE.flatten(), bins=50, alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title(f'I->E Weight Distribution\n(mean={W_IE.mean():.4f}, std={W_IE.std():.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict, by_experiment: Dict, output_dir: Path) -> str:
    """Generate markdown report."""

    # Compute summary statistics
    n_experiments = len(results)
    n_successful = sum(1 for r in results.values() if 'error' not in r)

    lines = [
        "# Input Restructuring Experiments Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "These experiments test whether SC factor selectivity arises from:",
        "1. **Direct input drive** - neurons receive factor information directly",
        "2. **Recurrent dynamics** - selectivity emerges from network interactions",
        "",
        f"**Total experiments:** {n_experiments}",
        f"**Successful:** {n_successful}",
        "",
        "## Experimental Conditions",
        "",
        "| Experiment | Description | Seeds |",
        "|------------|-------------|-------|",
    ]

    exp_descriptions = {
        'E0': 'Baseline: all neurons, all factors, at target onset',
        'E1': 'Temporal fix: spatial factors from trial start',
        'E2': 'Primary: RF-matched spatial (E only), sparse features (p=0.5)',
        'E3a': 'Sparsity test: p=0.25',
        'E3b': 'Sparsity test: p=0.75',
        'E4': 'Replication with different seeds',
    }

    for exp_type in ['E0', 'E1', 'E2', 'E3a', 'E3b', 'E4']:
        if exp_type in by_experiment:
            n_seeds = len(by_experiment[exp_type])
            lines.append(f"| {exp_type} | {exp_descriptions.get(exp_type, '')} | {n_seeds} |")

    lines.extend([
        "",
        "## Fit Quality",
        "",
    ])

    # Fit quality table
    lines.append("| Experiment | Mean Val Corr | Std | vs E0 |")
    lines.append("|------------|---------------|-----|-------|")

    e0_corr = None
    for exp_type in ['E0', 'E1', 'E2', 'E3a', 'E3b', 'E4']:
        if exp_type not in by_experiment:
            continue

        runs = by_experiment[exp_type]
        corrs = [r['val_corr'] for r in runs]
        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs) if len(corrs) > 1 else 0

        if exp_type == 'E0':
            e0_corr = mean_corr
            vs_e0 = '100%'
        elif e0_corr:
            vs_e0 = f'{mean_corr/e0_corr*100:.1f}%'
        else:
            vs_e0 = 'N/A'

        lines.append(f"| {exp_type} | {mean_corr:.4f} | {std_corr:.4f} | {vs_e0} |")

    lines.extend([
        "",
        "**Success criterion:** E2 achieves >= 80% of E0's correlation",
        "",
    ])

    if e0_corr and 'E2' in by_experiment:
        e2_corrs = [r['val_corr'] for r in by_experiment['E2']]
        e2_mean = np.mean(e2_corrs)
        if e2_mean >= 0.8 * e0_corr:
            lines.append(f"**PASSED:** E2 ({e2_mean:.4f}) >= 80% of E0 ({0.8*e0_corr:.4f})")
        else:
            lines.append(f"**FAILED:** E2 ({e2_mean:.4f}) < 80% of E0 ({0.8*e0_corr:.4f})")

    lines.extend([
        "",
        "## Emergent Selectivity",
        "",
        "Emergent selectivity = factor selectivity in neurons that do NOT receive that factor as input.",
        "",
    ])

    # Emergent selectivity table
    lines.append("| Factor | E2 Mean |sel|| E2 Frac Sig | Ablation Reduction |")
    lines.append("|--------|---------|-------------|---------------------|")

    if 'E2' in by_experiment:
        for factor in ['reward', 'probability', 'salience', 'identity']:
            runs = by_experiment['E2']
            mean_sels = []
            frac_sigs = []
            reductions = []

            for run in runs:
                sel = run.get('selectivity', {}).get(factor, {})
                mean_sels.append(sel.get('mean_abs_emergent', 0))
                frac_sigs.append(sel.get('fraction_significant', 0))

                abl = run.get('ablation', {}).get(factor, {}).get('full_recurrence', {})
                reductions.append(abl.get('reduction_fraction', 0))

            lines.append(f"| {factor} | {np.mean(mean_sels):.4f} | {np.mean(frac_sigs)*100:.1f}% | {np.mean(reductions)*100:.1f}% |")

    lines.extend([
        "",
        "## Key Findings",
        "",
        "### 1. Fit Quality Preserved?",
        "",
    ])

    if e0_corr and 'E2' in by_experiment:
        e2_corrs = [r['val_corr'] for r in by_experiment['E2']]
        e2_mean = np.mean(e2_corrs)
        ratio = e2_mean / e0_corr * 100
        lines.append(f"E2 achieves {ratio:.1f}% of E0 correlation.")
        if ratio >= 80:
            lines.append("**Result:** Input restructuring does NOT substantially degrade fit quality.")
        else:
            lines.append("**Result:** Input restructuring degrades fit quality. Consider adjusting model capacity.")

    lines.extend([
        "",
        "### 2. Emergent Selectivity Detected?",
        "",
    ])

    if 'E2' in by_experiment:
        runs = by_experiment['E2']
        any_significant = False
        for factor in ['reward', 'probability', 'salience', 'identity']:
            fracs = [r.get('selectivity', {}).get(factor, {}).get('fraction_significant', 0) for r in runs]
            if np.mean(fracs) > 0.1:
                any_significant = True
                lines.append(f"- **{factor}:** {np.mean(fracs)*100:.1f}% of non-receivers show significant selectivity")

        if any_significant:
            lines.append("")
            lines.append("**Result:** Yes, neurons show selectivity for factors they don't receive directly.")
        else:
            lines.append("**Result:** Limited emergent selectivity detected.")

    lines.extend([
        "",
        "### 3. Recurrence Dependence?",
        "",
    ])

    if 'E2' in by_experiment:
        runs = by_experiment['E2']
        for factor in ['reward', 'salience']:
            reductions = []
            for run in runs:
                abl = run.get('ablation', {}).get(factor, {}).get('full_recurrence', {})
                reductions.append(abl.get('reduction_fraction', 0))

            if reductions:
                mean_red = np.mean(reductions) * 100
                lines.append(f"- **{factor}:** {mean_red:.1f}% reduction after full recurrence ablation")

        lines.append("")
        lines.append("**Target:** >= 50% reduction confirms recurrence dependence.")

    lines.extend([
        "",
        "### 4. Cross-Seed Consistency?",
        "",
    ])

    if 'E2' in by_experiment and len(by_experiment['E2']) >= 2:
        runs = by_experiment['E2']
        ie_weights = []

        for run in runs:
            W_rec = run.get('W_rec')
            if W_rec is not None:
                n_total = W_rec.shape[0]
                n_exc = int(0.8 * n_total)
                ie_weights.append(W_rec[:n_exc, n_exc:].flatten())

        if len(ie_weights) >= 2:
            # Compute pairwise correlations
            corrs = []
            for i in range(len(ie_weights)):
                for j in range(i+1, len(ie_weights)):
                    r = np.corrcoef(ie_weights[i], ie_weights[j])[0, 1]
                    corrs.append(r)

            mean_corr = np.mean(corrs)
            lines.append(f"Mean pairwise I->E weight correlation: r = {mean_corr:.3f}")
            lines.append("")
            if mean_corr >= 0.5:
                lines.append("**Result:** I->E weight structure is consistent across seeds (robust computation).")
            else:
                lines.append("**Result:** I->E weight structure varies across seeds.")

    lines.extend([
        "",
        "## Figures",
        "",
        "- `fit_quality_comparison.png` - Validation correlation across experiments",
        "- `emergent_selectivity_by_factor.png` - Fraction significant per factor",
        "- `ablation_results.png` - Selectivity before/after ablation",
        "- `cross_seed_consistency.png` - I->E weight correlation across seeds",
        "- `ie_weight_structure.png` - I->E weight heatmap and distribution",
        "",
        "## Interpretation",
        "",
        "If emergent selectivity is detected AND ablation substantially reduces it,",
        "this supports the hypothesis that factor selectivity arises from local",
        "recurrent dynamics involving interneurons, rather than direct input drive.",
        "",
        "## Next Steps",
        "",
        "1. Compare I->E weight structure to E neuron factor selectivity",
        "2. Test whether specific I->E connection patterns predict selectivity",
        "3. Validate findings with additional sessions/animals",
        "",
    ])

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze input restructuring results')
    parser.add_argument('--results-dir', type=str,
                       default='results/input_restructuring',
                       help='Results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("INPUT RESTRUCTURING ANALYSIS")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print()

    # Load results
    print("Loading results...")
    results = load_all_results(results_dir)
    print(f"  Found {len(results)} experiment runs")

    by_experiment = extract_experiment_metrics(results)
    print(f"  Experiments: {list(by_experiment.keys())}")

    # Generate figures
    print("\nGenerating figures...")

    plot_fit_quality_comparison(by_experiment, analysis_dir / 'fit_quality_comparison.png')
    plot_emergent_selectivity_by_factor(by_experiment, analysis_dir / 'emergent_selectivity_by_factor.png')
    plot_ablation_results(by_experiment, analysis_dir / 'ablation_results.png')
    plot_cross_seed_consistency(by_experiment, analysis_dir / 'cross_seed_consistency.png')
    plot_ie_weight_structure(by_experiment, analysis_dir / 'ie_weight_structure.png')

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, by_experiment, results_dir)

    report_path = results_dir / 'REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Figures: {analysis_dir}")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
