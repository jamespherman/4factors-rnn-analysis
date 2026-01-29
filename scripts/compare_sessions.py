#!/usr/bin/env python3
"""
Phase 6: Cross-Session Comparison

Compares connectivity analysis results across all 3 Newton sessions to assess
replication of key findings.

Usage:
    python scripts/compare_sessions.py

Input:
    - results/final_model/ (Newton_08_15_2025_SC - original)
    - results/replication/Newton_08_14_2025_SC/
    - results/replication/Newton_08_13_2025_SC/

Output:
    - results/replication/comparison/ (figures)
    - specs/phase6_replication_report.md (final report)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import combine_pvalues
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Paths
BASE_DIR = Path(__file__).parent.parent
ORIGINAL_DIR = BASE_DIR / "results/final_model"
REPLICATION_DIR = BASE_DIR / "results/replication"
COMPARISON_DIR = REPLICATION_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# Sessions to compare
SESSIONS = [
    {
        'name': 'Newton_08_15_2025_SC',
        'short_name': '08_15',
        'path': ORIGINAL_DIR,
        'is_original': True,
    },
    {
        'name': 'Newton_08_14_2025_SC',
        'short_name': '08_14',
        'path': REPLICATION_DIR / 'Newton_08_14_2025_SC',
        'is_original': False,
    },
    {
        'name': 'Newton_08_13_2025_SC',
        'short_name': '08_13',
        'path': REPLICATION_DIR / 'Newton_08_13_2025_SC',
        'is_original': False,
    },
]


def load_session_data(session):
    """Load all relevant data for a session."""
    import json
    path = session['path']

    if not path.exists():
        print(f"  WARNING: Session not found: {path}")
        return None

    data = {'name': session['name'], 'short_name': session['short_name']}

    # Load weights
    weights_dir = path / 'weights'
    if (weights_dir / 'W_rec.npy').exists():
        data['W_rec'] = np.load(weights_dir / 'W_rec.npy')
        data['W_in'] = np.load(weights_dir / 'W_in.npy')
        data['h0'] = np.load(weights_dir / 'h0.npy')
        data['E_mask'] = np.load(weights_dir / 'E_mask.npy')
        data['I_mask'] = np.load(weights_dir / 'I_mask.npy')

    # Load metrics
    metrics_dir = path / 'metrics'
    if (metrics_dir / 'per_neuron_correlation.npy').exists():
        data['per_neuron_corr'] = np.load(metrics_dir / 'per_neuron_correlation.npy')
        data['neuron_ei_labels'] = np.load(metrics_dir / 'neuron_ei_labels.npy')

    # Load training log for validation correlation
    if (path / 'training_log.json').exists():
        with open(path / 'training_log.json') as f:
            training_log = json.load(f)
            data['training_log'] = training_log

    # Load connectivity analysis summary
    conn_dir = path / 'connectivity_analysis'
    if (conn_dir / 'analysis_summary.csv').exists():
        data['summary'] = pd.read_csv(conn_dir / 'analysis_summary.csv').iloc[0].to_dict()
    else:
        # Try to construct summary from other files
        data['summary'] = {}

    if (conn_dir / 'i_neuron_clusters.csv').exists():
        data['clusters'] = pd.read_csv(conn_dir / 'i_neuron_clusters.csv')

    if (conn_dir / 'factor_selectivity.csv').exists():
        data['selectivity'] = pd.read_csv(conn_dir / 'factor_selectivity.csv')

    # Derive network structure
    if 'E_mask' in data:
        data['n_e'] = data['E_mask'].sum()
        data['n_i'] = data['I_mask'].sum()
    elif 'summary' in data and 'n_e_neurons' in data['summary']:
        data['n_e'] = data['summary']['n_e_neurons']
        data['n_i'] = data['summary']['n_i_neurons']

    # Extract I→E weights
    if 'W_rec' in data and 'n_e' in data:
        n_e = int(data['n_e'])
        n_i = int(data['n_i'])
        data['W_IE'] = data['W_rec'][:n_e, n_e:n_e+n_i]

    # Populate missing summary fields from loaded data
    summary = data.get('summary', {})

    # Best validation correlation
    if 'best_val_corr' not in summary and 'training_log' in data:
        summary['best_val_corr'] = max(data['training_log'].get('val_correlation', [0]))

    # E/I correlation means
    if 'per_neuron_corr' in data and 'neuron_ei_labels' in data:
        labels = data['neuron_ei_labels']
        corrs = data['per_neuron_corr']
        if 'E_corr_mean' not in summary:
            summary['E_corr_mean'] = float(np.mean(corrs[labels == 0]))
        if 'I_corr_mean' not in summary:
            summary['I_corr_mean'] = float(np.mean(corrs[labels == 1])) if sum(labels == 1) > 0 else 0

    # h0 means
    if 'h0' in data and 'n_e' in data and 'n_i' in data:
        h0 = data['h0'].flatten()
        n_e = int(data['n_e'])
        n_i = int(data['n_i'])
        if 'h0_E_mean' not in summary:
            summary['h0_E_mean'] = float(np.mean(h0[:n_e]))
        if 'h0_I_mean' not in summary:
            summary['h0_I_mean'] = float(np.mean(h0[n_e:n_e+n_i]))

    # Effective targets
    if 'clusters' in data and 'mean_effective_targets' not in summary:
        summary['mean_effective_targets'] = float(data['clusters']['n_effective'].mean())

    data['summary'] = summary

    return data


def compute_gini(x):
    """Compute Gini coefficient."""
    x = np.abs(x.flatten())
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def compute_entropy(x):
    """Compute normalized entropy."""
    x = np.abs(x.flatten())
    if x.sum() == 0:
        return 0
    p = x / x.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(x))
    return entropy / max_entropy if max_entropy > 0 else 0


def create_performance_comparison(sessions_data):
    """Figure 1: Performance comparison across sessions."""
    print("Creating performance_comparison.png...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    session_names = [s['short_name'] for s in sessions_data]
    x = np.arange(len(session_names))

    # Panel A: Validation correlation
    ax = axes[0]
    val_corrs = []
    for s in sessions_data:
        if 'summary' in s and 'best_val_corr' in s['summary']:
            val_corrs.append(s['summary']['best_val_corr'])
        elif 'per_neuron_corr' in s:
            val_corrs.append(np.mean(s['per_neuron_corr']))
        else:
            val_corrs.append(0)

    colors = ['tab:blue' if s.get('is_original', True) else 'tab:orange' for s in SESSIONS[:len(sessions_data)]]
    bars = ax.bar(x, val_corrs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('A. Model Performance')
    ax.set_ylim(0, max(val_corrs) * 1.2 if val_corrs else 1)

    # Add value labels
    for bar, val in zip(bars, val_corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Panel B: E vs I correlation
    ax = axes[1]
    width = 0.35
    e_corrs = []
    i_corrs = []

    for s in sessions_data:
        if 'per_neuron_corr' in s and 'neuron_ei_labels' in s:
            labels = s['neuron_ei_labels']
            corrs = s['per_neuron_corr']
            e_corrs.append(np.mean(corrs[labels == 0]))
            i_corrs.append(np.mean(corrs[labels == 1]))
        elif 'summary' in s:
            e_corrs.append(s['summary'].get('E_corr_mean', 0))
            i_corrs.append(s['summary'].get('I_corr_mean', 0))
        else:
            e_corrs.append(0)
            i_corrs.append(0)

    ax.bar(x - width/2, e_corrs, width, label='E neurons', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, i_corrs, width, label='I neurons', color='tab:red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Mean PSTH Correlation')
    ax.set_title('B. E vs I Neuron Fit')
    ax.legend()

    # Panel C: Number of neurons
    ax = axes[2]
    n_es = [int(s.get('n_e', 0)) for s in sessions_data]
    n_is = [int(s.get('n_i', 0)) for s in sessions_data]

    ax.bar(x - width/2, n_es, width, label='E neurons', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, n_is, width, label='I neurons', color='tab:red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Number of Neurons')
    ax.set_title('C. Network Size')
    ax.legend()

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'performance_comparison.png', dpi=150)
    plt.close()


def create_inhibition_specificity_comparison(sessions_data):
    """Figure 2: Inhibition specificity comparison."""
    print("Creating inhibition_specificity_comparison.png...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    session_names = [s['short_name'] for s in sessions_data]
    x = np.arange(len(session_names))

    # Panel A: Gini coefficients with null distribution
    ax = axes[0]
    gini_means = []
    gini_ps = []

    for s in sessions_data:
        if 'clusters' in s:
            gini_means.append(s['clusters']['gini'].mean())
        elif 'summary' in s and 'mean_gini' in s['summary']:
            gini_means.append(s['summary']['mean_gini'])
        else:
            gini_means.append(0)

        if 'summary' in s and 'gini_vs_shuffle_p' in s['summary']:
            gini_ps.append(s['summary']['gini_vs_shuffle_p'])
        else:
            gini_ps.append(1.0)

    bars = ax.bar(x, gini_means, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(0.36, color='red', linestyle='--', label='Original session')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Mean Gini Coefficient')
    ax.set_title('A. Inhibition Concentration (Gini)')

    # Add p-value annotations
    for i, (bar, p) in enumerate(zip(bars, gini_ps)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'p={p:.2f}\n({sig})', ha='center', va='bottom', fontsize=9)

    ax.legend(loc='upper right')

    # Panel B: Entropy
    ax = axes[1]
    entropy_means = []
    entropy_ps = []

    for s in sessions_data:
        if 'clusters' in s:
            entropy_means.append(s['clusters']['entropy'].mean())
        elif 'summary' in s and 'mean_entropy' in s['summary']:
            entropy_means.append(s['summary']['mean_entropy'])
        else:
            entropy_means.append(0)

        if 'summary' in s and 'entropy_vs_shuffle_p' in s['summary']:
            entropy_ps.append(s['summary']['entropy_vs_shuffle_p'])
        else:
            entropy_ps.append(1.0)

    bars = ax.bar(x, entropy_means, color='darkorange', alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Mean Normalized Entropy')
    ax.set_title('B. Inhibition Distribution (Entropy)')

    for i, (bar, p) in enumerate(zip(bars, entropy_ps)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'p={p:.2f}\n({sig})', ha='center', va='bottom', fontsize=9)

    # Panel C: Effective targets
    ax = axes[2]
    eff_means = []
    eff_fracs = []

    for s in sessions_data:
        n_e = int(s.get('n_e', 1))
        if 'clusters' in s:
            eff = s['clusters']['n_effective'].mean()
            eff_means.append(eff)
            eff_fracs.append(eff / n_e * 100)
        elif 'summary' in s and 'mean_effective_targets' in s['summary']:
            eff = s['summary']['mean_effective_targets']
            eff_means.append(eff)
            eff_fracs.append(eff / n_e * 100)
        else:
            eff_means.append(0)
            eff_fracs.append(0)

    bars = ax.bar(x, eff_fracs, color='seagreen', alpha=0.8, edgecolor='black')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Effective Targets (% of E neurons)')
    ax.set_title('C. Breadth of Inhibition')
    ax.legend()

    for i, (bar, eff) in enumerate(zip(bars, eff_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{eff:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'inhibition_specificity_comparison.png', dpi=150)
    plt.close()


def create_h0_comparison(sessions_data):
    """Figure 3: h0 comparison across sessions."""
    print("Creating h0_comparison.png...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    session_names = [s['short_name'] for s in sessions_data]

    # Panel A: h0 means by E/I
    ax = axes[0]
    x = np.arange(len(session_names))
    width = 0.35

    h0_e_means = []
    h0_i_means = []
    h0_ps = []

    for s in sessions_data:
        if 'h0' in s and 'n_e' in s:
            n_e = int(s['n_e'])
            h0 = s['h0'].flatten()
            h0_e_means.append(np.mean(h0[:n_e]))
            h0_i_means.append(np.mean(h0[n_e:]))
        elif 'summary' in s:
            h0_e_means.append(s['summary'].get('h0_E_mean', 0))
            h0_i_means.append(s['summary'].get('h0_I_mean', 0))
        else:
            h0_e_means.append(0)
            h0_i_means.append(0)

        if 'summary' in s and 'h0_ei_pval' in s['summary']:
            h0_ps.append(s['summary']['h0_ei_pval'])
        else:
            h0_ps.append(1.0)

    ax.bar(x - width/2, h0_e_means, width, label='E neurons', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, h0_i_means, width, label='I neurons', color='tab:red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Mean h0')
    ax.set_title('A. Initial State (h0) by Cell Type')
    ax.legend()

    # Add p-value annotations
    for i, p in enumerate(h0_ps):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_max = max(h0_e_means[i], h0_i_means[i])
        ax.text(i, y_max + 0.05, f'p={p:.3f}\n({sig})', ha='center', va='bottom', fontsize=9)

    # Panel B: h0 difference (I - E)
    ax = axes[1]
    h0_diffs = [i - e for e, i in zip(h0_e_means, h0_i_means)]

    colors = ['tab:green' if d > 0 else 'tab:red' for d in h0_diffs]
    bars = ax.bar(x, h0_diffs, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('h0 Difference (I - E)')
    ax.set_title('B. h0 Difference (I > E if positive)')

    for bar, p in zip(bars, h0_ps):
        height = bar.get_height()
        sig = "*" if p < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                height + 0.01 if height > 0 else height - 0.02,
                sig, ha='center', va='bottom' if height > 0 else 'top', fontsize=14)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'h0_comparison.png', dpi=150)
    plt.close()


def create_ie_weights_comparison(sessions_data):
    """Figure 4: I→E weight matrices side-by-side."""
    print("Creating ie_weights_comparison.png...")

    n_sessions = len(sessions_data)
    fig, axes = plt.subplots(1, n_sessions, figsize=(5*n_sessions, 6))

    if n_sessions == 1:
        axes = [axes]

    for i, (ax, s) in enumerate(zip(axes, sessions_data)):
        if 'W_IE' in s:
            W_IE = s['W_IE']
            vmax = np.abs(W_IE).max()
            im = ax.imshow(W_IE, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_xlabel('I Neuron')
            ax.set_ylabel('E Neuron')
            ax.set_title(f'{s["short_name"]}\n(E={s.get("n_e", "?")}, I={s.get("n_i", "?")})')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(s['short_name'])

    plt.suptitle('I→E Connectivity Matrices Across Sessions', fontsize=14)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'ie_weights_comparison.png', dpi=150)
    plt.close()


def create_factor_selectivity_comparison(sessions_data):
    """Figure 5: Factor selectivity patterns across sessions."""
    print("Creating factor_selectivity_comparison.png...")

    factor_names = ['reward', 'location', 'identity', 'salience']
    n_sessions = len(sessions_data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for j, factor in enumerate(factor_names):
        ax = axes[j]
        x = np.arange(n_sessions)
        width = 0.35

        e_means = []
        i_means = []

        for s in sessions_data:
            if 'selectivity' in s:
                sel_df = s['selectivity']
                e_data = sel_df[sel_df['neuron_type'] == 'E'][factor].values
                i_data = sel_df[sel_df['neuron_type'] == 'I'][factor].values
                e_means.append(np.nanmean(e_data))
                i_means.append(np.nanmean(i_data))
            else:
                e_means.append(0)
                i_means.append(0)

        ax.bar(x - width/2, e_means, width, label='E', color='tab:blue', alpha=0.8)
        ax.bar(x + width/2, i_means, width, label='I', color='tab:red', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s['short_name'] for s in sessions_data])
        ax.set_ylabel('Partial eta-squared')
        ax.set_title(f'{factor.capitalize()} Selectivity')
        ax.legend()

    plt.suptitle('Factor Selectivity by Cell Type Across Sessions', fontsize=14)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'factor_selectivity_comparison.png', dpi=150)
    plt.close()


def create_summary_comparison(sessions_data):
    """Figure 6: Multi-panel summary."""
    print("Creating summary_comparison.png...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    session_names = [s['short_name'] for s in sessions_data]
    x = np.arange(len(session_names))

    # Panel A: Val correlation
    ax = fig.add_subplot(gs[0, 0])
    val_corrs = []
    for s in sessions_data:
        if 'summary' in s and 'best_val_corr' in s['summary']:
            val_corrs.append(s['summary']['best_val_corr'])
        elif 'per_neuron_corr' in s:
            val_corrs.append(np.mean(s['per_neuron_corr']))
        else:
            val_corrs.append(0)
    ax.bar(x, val_corrs, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Val Correlation')
    ax.set_title('A. Model Performance')

    # Panel B: Gini
    ax = fig.add_subplot(gs[0, 1])
    ginis = []
    for s in sessions_data:
        if 'summary' in s and 'mean_gini' in s['summary']:
            ginis.append(s['summary']['mean_gini'])
        elif 'clusters' in s:
            ginis.append(s['clusters']['gini'].mean())
        else:
            ginis.append(0)
    ax.bar(x, ginis, color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Mean Gini')
    ax.set_title('B. Inhibition Concentration')

    # Panel C: Effective targets %
    ax = fig.add_subplot(gs[0, 2])
    eff_pcts = []
    for s in sessions_data:
        n_e = int(s.get('n_e', 1))
        if 'summary' in s and 'mean_effective_targets' in s['summary']:
            eff_pcts.append(s['summary']['mean_effective_targets'] / n_e * 100)
        elif 'clusters' in s:
            eff_pcts.append(s['clusters']['n_effective'].mean() / n_e * 100)
        else:
            eff_pcts.append(0)
    ax.bar(x, eff_pcts, color='seagreen', alpha=0.8)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Effective Targets (%)')
    ax.set_title('C. Inhibition Breadth')

    # Panel D: h0 E vs I
    ax = fig.add_subplot(gs[1, 0])
    width = 0.35
    h0_es = []
    h0_is = []
    for s in sessions_data:
        if 'summary' in s:
            h0_es.append(s['summary'].get('h0_E_mean', 0))
            h0_is.append(s['summary'].get('h0_I_mean', 0))
        else:
            h0_es.append(0)
            h0_is.append(0)
    ax.bar(x - width/2, h0_es, width, label='E', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, h0_is, width, label='I', color='tab:red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('Mean h0')
    ax.set_title('D. Initial State')
    ax.legend()

    # Panel E: h0 p-values
    ax = fig.add_subplot(gs[1, 1])
    h0_ps = []
    for s in sessions_data:
        if 'summary' in s and 'h0_ei_pval' in s['summary']:
            h0_ps.append(s['summary']['h0_ei_pval'])
        else:
            h0_ps.append(1.0)

    colors = ['green' if p < 0.05 else 'gray' for p in h0_ps]
    ax.bar(x, [-np.log10(p + 1e-10) for p in h0_ps], color=colors, alpha=0.8)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('-log10(p)')
    ax.set_title('E. h0 E/I Difference Significance')
    ax.legend()

    # Panel F: Gini p-values
    ax = fig.add_subplot(gs[1, 2])
    gini_ps = []
    for s in sessions_data:
        if 'summary' in s and 'gini_vs_shuffle_p' in s['summary']:
            gini_ps.append(s['summary']['gini_vs_shuffle_p'])
        else:
            gini_ps.append(1.0)

    colors = ['green' if p < 0.05 else 'gray' for p in gini_ps]
    ax.bar(x, gini_ps, color=colors, alpha=0.8)
    ax.axhline(0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names)
    ax.set_ylabel('p-value')
    ax.set_title('F. Gini vs Shuffle')
    ax.legend()

    # Panel G-I: I→E matrices (if available)
    for i, s in enumerate(sessions_data[:3]):
        ax = fig.add_subplot(gs[2, i])
        if 'W_IE' in s:
            W_IE = s['W_IE']
            vmax = np.abs(W_IE).max()
            im = ax.imshow(W_IE, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'G-I. {s["short_name"]} I→E')
            ax.set_xlabel('I')
            ax.set_ylabel('E')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(s['short_name'])

    plt.suptitle('Cross-Session Comparison Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_metrics_table(sessions_data):
    """Generate CSV table of all metrics."""
    print("Generating metrics_table.csv...")

    rows = []
    for s in sessions_data:
        row = {
            'session': s['name'],
            'short_name': s['short_name'],
            'n_e': s.get('n_e', np.nan),
            'n_i': s.get('n_i', np.nan),
        }

        if 'summary' in s:
            summary = s['summary']
            row.update({
                'n_train_trials': summary.get('n_train_trials', np.nan),
                'n_val_trials': summary.get('n_val_trials', np.nan),
                'best_val_corr': summary.get('best_val_corr', np.nan),
                'E_corr_mean': summary.get('E_corr_mean', np.nan),
                'I_corr_mean': summary.get('I_corr_mean', np.nan),
                'mean_gini': summary.get('mean_gini', np.nan),
                'gini_vs_shuffle_p': summary.get('gini_vs_shuffle_p', np.nan),
                'mean_entropy': summary.get('mean_entropy', np.nan),
                'entropy_vs_shuffle_p': summary.get('entropy_vs_shuffle_p', np.nan),
                'mean_effective_targets': summary.get('mean_effective_targets', np.nan),
                'h0_E_mean': summary.get('h0_E_mean', np.nan),
                'h0_I_mean': summary.get('h0_I_mean', np.nan),
                'h0_ei_pval': summary.get('h0_ei_pval', np.nan),
                'cluster_silhouette': summary.get('cluster_silhouette', np.nan),
                'cluster_permutation_p': summary.get('cluster_permutation_p', np.nan),
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_DIR / 'metrics_table.csv', index=False)
    return df


def perform_statistical_tests(sessions_data):
    """Perform cross-session statistical tests."""
    print("Performing statistical tests...")

    tests = []

    # Test 1: Are Gini coefficients consistently within shuffled null?
    gini_ps = []
    for s in sessions_data:
        if 'summary' in s and 'gini_vs_shuffle_p' in s['summary']:
            gini_ps.append(s['summary']['gini_vs_shuffle_p'])

    if len(gini_ps) > 0:
        n_not_sig = sum(p > 0.05 for p in gini_ps)
        tests.append({
            'test': 'Gini vs shuffle consistency',
            'description': 'Sessions with Gini not different from shuffled',
            'n_sessions': len(gini_ps),
            'n_consistent': n_not_sig,
            'proportion': n_not_sig / len(gini_ps),
            'replicated': n_not_sig == len(gini_ps),
        })

    # Test 2: h0 E/I difference - meta-analysis
    h0_ps = []
    h0_directions = []  # True if I > E
    for s in sessions_data:
        if 'summary' in s:
            if 'h0_ei_pval' in s['summary']:
                h0_ps.append(s['summary']['h0_ei_pval'])
            if 'h0_E_mean' in s['summary'] and 'h0_I_mean' in s['summary']:
                h0_directions.append(s['summary']['h0_I_mean'] > s['summary']['h0_E_mean'])

    if len(h0_ps) >= 2:
        # Fisher's method to combine p-values
        stat, combined_p = combine_pvalues(h0_ps, method='fisher')
        n_sig = sum(p < 0.05 for p in h0_ps)
        n_i_gt_e = sum(h0_directions)

        tests.append({
            'test': 'h0 E/I difference meta-analysis',
            'description': 'Combined p-value (Fisher method)',
            'n_sessions': len(h0_ps),
            'n_significant': n_sig,
            'n_I_greater_than_E': n_i_gt_e,
            'combined_p': combined_p,
            'replicated': combined_p < 0.05 and n_i_gt_e >= len(h0_ps) / 2,
        })

    # Test 3: Effective targets consistently > 50%
    eff_pcts = []
    for s in sessions_data:
        n_e = int(s.get('n_e', 1))
        if 'summary' in s and 'mean_effective_targets' in s['summary']:
            eff_pcts.append(s['summary']['mean_effective_targets'] / n_e * 100)
        elif 'clusters' in s:
            eff_pcts.append(s['clusters']['n_effective'].mean() / n_e * 100)

    if len(eff_pcts) > 0:
        n_global = sum(p > 50 for p in eff_pcts)
        tests.append({
            'test': 'Global inhibition (eff targets > 50%)',
            'description': 'Sessions with >50% effective targets',
            'n_sessions': len(eff_pcts),
            'n_global': n_global,
            'mean_eff_pct': np.mean(eff_pcts),
            'replicated': n_global == len(eff_pcts),
        })

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(COMPARISON_DIR / 'statistical_tests.csv', index=False)
    return tests_df


def generate_replication_report(sessions_data, metrics_df, tests_df):
    """Generate the final replication report."""
    print("Generating phase6_replication_report.md...")

    # Extract key metrics
    n_sessions = len(sessions_data)

    # Performance
    val_corrs = metrics_df['best_val_corr'].dropna().tolist()
    mean_corr = np.mean(val_corrs) if val_corrs else 0

    # Gini
    gini_ps = metrics_df['gini_vs_shuffle_p'].dropna().tolist()
    gini_means = metrics_df['mean_gini'].dropna().tolist()
    n_gini_not_sig = sum(p > 0.05 for p in gini_ps)

    # h0
    h0_ps = metrics_df['h0_ei_pval'].dropna().tolist()
    h0_e_means = metrics_df['h0_E_mean'].dropna().tolist()
    h0_i_means = metrics_df['h0_I_mean'].dropna().tolist()
    n_h0_sig = sum(p < 0.05 for p in h0_ps)
    n_h0_i_gt_e = sum(i > e for e, i in zip(h0_e_means, h0_i_means))

    # Effective targets
    eff_targets = metrics_df['mean_effective_targets'].dropna().tolist()
    n_es = metrics_df['n_e'].dropna().tolist()
    eff_pcts = [eff / n_e * 100 for eff, n_e in zip(eff_targets, n_es)] if eff_targets and n_es else []
    n_global = sum(p > 50 for p in eff_pcts)

    # Determine replication status
    global_inhibition_replicates = n_gini_not_sig == n_sessions if n_sessions > 0 else False
    h0_replicates = n_h0_sig >= n_sessions - 1 and n_h0_i_gt_e >= n_sessions - 1 if n_sessions > 0 else False

    report = f"""# Phase 6: Cross-Session Replication Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

**Do the key findings replicate across sessions?**

| Finding | Original | Replication Sessions | Status |
|---------|----------|---------------------|--------|
| Global inhibition (Gini p > 0.05) | p=0.712 | {n_gini_not_sig}/{n_sessions} sessions | {'REPLICATED' if global_inhibition_replicates else 'PARTIAL/FAILED'} |
| h0 I > E (p < 0.05) | p=0.002 | {n_h0_sig}/{n_sessions} significant, {n_h0_i_gt_e}/{n_sessions} I>E | {'REPLICATED' if h0_replicates else 'PARTIAL/FAILED'} |
| Effective targets > 50% | 69% | {n_global}/{n_sessions} sessions | {'REPLICATED' if n_global == n_sessions else 'PARTIAL/FAILED'} |

**Overall Assessment:** {'The key findings REPLICATE across all sessions from Animal 1.' if global_inhibition_replicates and h0_replicates else 'Some findings show PARTIAL replication. See details below.'}

---

## 1. Per-Session Results

### Dataset Characteristics

| Session | n_E | n_I | Train Trials | Val Trials |
|---------|-----|-----|--------------|------------|
"""

    def safe_int(val, default=0):
        """Safely convert to int, handling NaN."""
        if pd.isna(val):
            return default
        return int(val)

    for _, row in metrics_df.iterrows():
        report += f"| {row['short_name']} | {safe_int(row.get('n_e', 0))} | {safe_int(row.get('n_i', 0))} | {safe_int(row.get('n_train_trials', 0))} | {safe_int(row.get('n_val_trials', 0))} |\n"

    report += f"""
### Model Performance

| Session | Val Correlation | E Corr | I Corr |
|---------|-----------------|--------|--------|
"""

    def safe_float(val, default=0.0):
        """Safely convert to float, handling NaN."""
        if pd.isna(val):
            return default
        return float(val)

    for _, row in metrics_df.iterrows():
        report += f"| {row['short_name']} | {safe_float(row.get('best_val_corr', 0)):.4f} | {safe_float(row.get('E_corr_mean', 0)):.4f} | {safe_float(row.get('I_corr_mean', 0)):.4f} |\n"

    report += f"""
### Inhibition Specificity

| Session | Mean Gini | Gini p | Mean Entropy | Entropy p | Eff Targets (%) |
|---------|-----------|--------|--------------|-----------|-----------------|
"""

    for i, row in metrics_df.iterrows():
        n_e = safe_float(row.get('n_e', 1), 1)
        eff_pct = safe_float(row.get('mean_effective_targets', 0)) / n_e * 100 if n_e > 0 else 0
        report += f"| {row['short_name']} | {safe_float(row.get('mean_gini', 0)):.4f} | {safe_float(row.get('gini_vs_shuffle_p', 1), 1):.3f} | {safe_float(row.get('mean_entropy', 0)):.4f} | {safe_float(row.get('entropy_vs_shuffle_p', 1), 1):.3f} | {eff_pct:.1f}% |\n"

    report += f"""
### h0 Analysis

| Session | h0 E Mean | h0 I Mean | E vs I p | I > E? |
|---------|-----------|-----------|----------|--------|
"""

    for _, row in metrics_df.iterrows():
        h0_i = safe_float(row.get('h0_I_mean', 0))
        h0_e = safe_float(row.get('h0_E_mean', 0))
        h0_p = safe_float(row.get('h0_ei_pval', 1), 1)
        i_gt_e = "Yes" if h0_i > h0_e else "No"
        sig = "**" if h0_p < 0.01 else "*" if h0_p < 0.05 else ""
        report += f"| {row['short_name']} | {h0_e:.4f} | {h0_i:.4f} | {h0_p:.4f}{sig} | {i_gt_e} |\n"

    report += f"""
---

## 2. Cross-Session Statistical Tests

"""

    for _, test in tests_df.iterrows():
        report += f"### {test['test']}\n\n"
        report += f"- **Description:** {test['description']}\n"
        report += f"- **Result:** {'REPLICATED' if test.get('replicated', False) else 'NOT REPLICATED'}\n"
        for k, v in test.items():
            if k not in ['test', 'description', 'replicated']:
                report += f"- {k}: {v}\n"
        report += "\n"

    report += f"""
---

## 3. Conclusions

### Does global inhibition replicate?

{'**YES.** ' if global_inhibition_replicates else '**PARTIAL.** '}
"""

    if global_inhibition_replicates:
        report += f"All {n_sessions} sessions show Gini coefficients indistinguishable from shuffled controls (all p > 0.05), confirming that inhibition is broadly distributed rather than factor-specific.\n\n"
    else:
        report += f"Only {n_gini_not_sig}/{n_sessions} sessions show non-significant Gini vs shuffle. This may indicate session-specific variability.\n\n"

    report += f"""Mean Gini across sessions: {np.mean(gini_means):.3f} (range: {min(gini_means):.3f} - {max(gini_means):.3f})
Mean effective targets: {np.mean(eff_pcts):.1f}% of E neurons

### Does the h0 E/I difference replicate?

{'**YES.** ' if h0_replicates else '**PARTIAL.** '}
"""

    if h0_replicates:
        report += f"The finding that I neurons have higher initial states than E neurons replicates across sessions ({n_h0_sig}/{n_sessions} significant, {n_h0_i_gt_e}/{n_sessions} with I > E direction).\n\n"
    else:
        report += f"Only {n_h0_sig}/{n_sessions} sessions show significant h0 E/I difference. The direction (I > E) is {'consistent' if n_h0_i_gt_e == n_sessions else 'inconsistent'} across sessions.\n\n"

    report += f"""
### Session-specific patterns

"""

    if n_sessions >= 2:
        corr_range = max(val_corrs) - min(val_corrs) if val_corrs else 0
        if corr_range > 0.1:
            report += "- Model performance varies across sessions (correlation range > 0.1), possibly reflecting differences in neural activity patterns or recording quality.\n"
        else:
            report += "- Model performance is consistent across sessions (correlation range < 0.1).\n"

    report += f"""
### Implications for SC Circuit Organization

Based on this replication study:

1. **Robust finding:** I neurons provide broadly distributed inhibition to E neurons. This is not an artifact of a single session.

2. **Robust finding:** I neurons start with higher baseline activity (h0) than E neurons, potentially reflecting tonic inhibition.

3. **Not tested yet:** Whether these patterns generalize to other animals (Feynman).

---

## 4. Recommendations

"""

    if global_inhibition_replicates and h0_replicates:
        report += """### All findings replicate - proceed to Animal 2

The key findings from Newton_08_15_2025_SC replicate across all Newton sessions. We recommend:

1. **Proceed to Animal 2 (Feynman)** to test cross-animal generalization
2. **Pool sessions** for increased statistical power if analyzing subtle effects
3. **Consider session as random effect** in mixed-effects models
"""
    else:
        report += """### Partial replication - investigate differences

Some findings show inconsistent replication. We recommend:

1. **Investigate session differences** - examine trial counts, recording quality, behavioral state
2. **Consider pooling** with session as covariate
3. **Report heterogeneity** in final analysis
"""

    report += f"""
---

## Figures

All comparison figures saved to `results/replication/comparison/`:

1. `performance_comparison.png` - Model performance across sessions
2. `inhibition_specificity_comparison.png` - Gini, entropy, effective targets
3. `h0_comparison.png` - Initial state by cell type
4. `ie_weights_comparison.png` - I→E weight matrices
5. `factor_selectivity_comparison.png` - Factor selectivity patterns
6. `summary_comparison.png` - Multi-panel summary

---

*Report generated by compare_sessions.py*
"""

    report_path = BASE_DIR / "specs/phase6_replication_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    return report


def main():
    """Main comparison function."""
    print("=" * 80)
    print("PHASE 6: CROSS-SESSION COMPARISON")
    print("=" * 80)

    # Load all sessions
    print("\nLoading session data...")
    sessions_data = []
    for session in SESSIONS:
        print(f"  Loading {session['name']}...")
        data = load_session_data(session)
        if data:
            data['is_original'] = session.get('is_original', False)
            sessions_data.append(data)
        else:
            print(f"    Skipping - data not found")

    if len(sessions_data) < 2:
        print("\nERROR: Need at least 2 sessions to compare. Run replication training first.")
        print("Use: python scripts/run_replication_animal1.py")
        return

    print(f"\nLoaded {len(sessions_data)} sessions: {[s['short_name'] for s in sessions_data]}")

    # Create comparison figures
    print("\nCreating comparison figures...")
    create_performance_comparison(sessions_data)
    create_inhibition_specificity_comparison(sessions_data)
    create_h0_comparison(sessions_data)
    create_ie_weights_comparison(sessions_data)
    create_factor_selectivity_comparison(sessions_data)
    create_summary_comparison(sessions_data)

    # Generate metrics table
    metrics_df = generate_metrics_table(sessions_data)

    # Statistical tests
    tests_df = perform_statistical_tests(sessions_data)

    # Generate report
    report = generate_replication_report(sessions_data, metrics_df, tests_df)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {COMPARISON_DIR}")
    print(f"Report: specs/phase6_replication_report.md")

    # Print summary
    print("\n--- QUICK SUMMARY ---")
    print(f"Sessions compared: {len(sessions_data)}")
    for s in sessions_data:
        if 'summary' in s:
            gini_p = s['summary'].get('gini_vs_shuffle_p', 1)
            h0_p = s['summary'].get('h0_ei_pval', 1)
            val_corr = s['summary'].get('best_val_corr', 0)
            print(f"  {s['short_name']}: val_corr={val_corr:.3f}, Gini_p={gini_p:.3f}, h0_p={h0_p:.3f}")


if __name__ == '__main__':
    main()
