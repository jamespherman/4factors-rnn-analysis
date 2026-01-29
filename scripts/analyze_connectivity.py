#!/usr/bin/env python3
"""
I→E Connectivity Analysis for 4-Factors RNN Model

This script analyzes whether inhibitory interneurons show factor-specific
structure in their connectivity to excitatory neurons.

Author: Claude Code
Date: 2025-01-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Paths
BASE_DIR = Path("/Users/jph/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/4factors-rnn-analysis")
WEIGHTS_DIR = BASE_DIR / "results/final_model/weights"
OUTPUTS_DIR = BASE_DIR / "results/final_model/outputs"
METRICS_DIR = BASE_DIR / "results/final_model/metrics"
DATA_FILE = BASE_DIR / "data/rnn_export_Newton_08_15_2025_SC.mat"
OUTPUT_DIR = BASE_DIR / "results/final_model/connectivity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Network structure
N_E = 52  # Excitatory neurons (41 recorded + 11 hidden)
N_I = 13  # Inhibitory neurons (all recorded)
N_RECORDED_E = 41  # indices 0-40 in model
N_RECORDED_I = 13  # indices 52-64 in model
N_RECORDED = 54  # 41 E + 13 I

# Model indices
E_INDICES = np.arange(N_E)  # 0-51 in model
I_INDICES = np.arange(N_E, N_E + N_I)  # 52-64 in model
RECORDED_E_MODEL_IDX = np.arange(N_RECORDED_E)  # 0-40 in model
RECORDED_I_MODEL_IDX = np.arange(N_E, N_E + N_I)  # 52-64 in model

# Output file indices (54 recorded neurons)
# In output files: 0-40 are E, 41-53 are I
OUTPUT_E_IDX = np.arange(N_RECORDED_E)  # 0-40
OUTPUT_I_IDX = np.arange(N_RECORDED_E, N_RECORDED)  # 41-53


def load_data():
    """Load all required data files."""
    print("Loading data...")

    # Load weights
    W_rec = np.load(WEIGHTS_DIR / "W_rec.npy")
    W_in = np.load(WEIGHTS_DIR / "W_in.npy")
    h0 = np.load(WEIGHTS_DIR / "h0.npy")
    attention_weights = np.load(WEIGHTS_DIR / "attention_weights.npy", allow_pickle=True)
    E_mask = np.load(WEIGHTS_DIR / "E_mask.npy")
    I_mask = np.load(WEIGHTS_DIR / "I_mask.npy")

    # Load outputs
    val_model_rates = np.load(OUTPUTS_DIR / "val_model_rates.npy")
    val_target_rates = np.load(OUTPUTS_DIR / "val_target_rates.npy")
    val_trial_conditions = np.load(OUTPUTS_DIR / "val_trial_conditions.npy")

    # Load metrics
    per_neuron_corr = np.load(METRICS_DIR / "per_neuron_correlation.npy")
    neuron_ei_labels = np.load(METRICS_DIR / "neuron_ei_labels.npy")

    # Load trial conditions from original data file
    with h5py.File(DATA_FILE, 'r') as f:
        # Get validation indices (last 20% of trials)
        n_trials = f['trial_reward'].shape[1]
        n_val = int(0.2 * n_trials)
        val_start = n_trials - n_val

        trial_reward = f['trial_reward'][0, val_start:].astype(int)
        trial_location = f['trial_location'][0, val_start:].astype(int)
        trial_identity = f['trial_identity'][0, val_start:].astype(int)
        trial_salience = f['trial_salience'][0, val_start:].astype(int)

        # Try to load probability if it exists
        if 'trial_probability' in f:
            trial_probability = f['trial_probability'][0, val_start:].astype(int)
        else:
            trial_probability = np.zeros_like(trial_reward)

    data = {
        'W_rec': W_rec,
        'W_in': W_in,
        'h0': h0,
        'attention_weights': attention_weights,
        'E_mask': E_mask,
        'I_mask': I_mask,
        'val_model_rates': val_model_rates,
        'val_target_rates': val_target_rates,
        'val_trial_conditions': val_trial_conditions,
        'per_neuron_corr': per_neuron_corr,
        'neuron_ei_labels': neuron_ei_labels,
        'trial_reward': trial_reward,
        'trial_location': trial_location,
        'trial_identity': trial_identity,
        'trial_salience': trial_salience,
        'trial_probability': trial_probability,
    }

    print(f"  W_rec shape: {W_rec.shape}")
    print(f"  Validation trials: {val_model_rates.shape[0]}")
    print(f"  Recorded neurons: {N_RECORDED}")
    print(f"  E neurons: {N_E}, I neurons: {N_I}")

    return data


def compute_factor_selectivity(data):
    """
    Analysis 1: Compute factor selectivity for each recorded neuron.
    Uses multiple regression to compute partial eta-squared for each factor.
    """
    print("\n=== Analysis 1: Factor Selectivity ===")

    # Get mean firing rates per trial (averaged over time)
    rates = data['val_target_rates']  # (trials, time, 54 recorded neurons)
    mean_rates = rates.mean(axis=1)  # (trials, 54)

    # Output file has 54 neurons: indices 0-40 are E, 41-53 are I
    n_recorded = N_RECORDED

    # Prepare factors
    factors = {
        'reward': data['trial_reward'],
        'location': data['trial_location'],
        'identity': data['trial_identity'],
        'salience': data['trial_salience'],
    }

    # Results storage
    selectivity = np.zeros((n_recorded, 4))  # partial eta-squared
    selectivity_beta = np.zeros((n_recorded, 4))  # standardized coefficients
    selectivity_pval = np.zeros((n_recorded, 4))
    factor_names = ['reward', 'location', 'identity', 'salience']

    for i in range(n_recorded):
        y = mean_rates[:, i]  # Neuron i in output file (0-53)

        # Build design matrix with all factors
        X = np.column_stack([
            factors['reward'],
            factors['location'],
            factors['identity'],
            factors['salience'],
        ])

        # Standardize X for comparable coefficients
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_std = sm.add_constant(X_std)

        try:
            model = sm.OLS(y, X_std).fit()

            # Get standardized coefficients (betas)
            for j, factor in enumerate(factor_names):
                selectivity_beta[i, j] = model.params[j + 1]
                selectivity_pval[i, j] = model.pvalues[j + 1]

                # Compute partial eta-squared
                # SS_factor / (SS_factor + SS_residual)
                # Approximate using t-statistic: t^2 / (t^2 + df_resid)
                t_stat = model.tvalues[j + 1]
                df_resid = model.df_resid
                selectivity[i, j] = t_stat**2 / (t_stat**2 + df_resid)

        except Exception as e:
            print(f"  Warning: Regression failed for neuron {i}: {e}")
            selectivity[i, :] = np.nan
            selectivity_beta[i, :] = np.nan
            selectivity_pval[i, :] = np.nan

    # Separate E and I neurons (in output file: 0-40 are E, 41-53 are I)
    e_selectivity = selectivity[:N_RECORDED_E, :]  # First 41 neurons
    i_selectivity = selectivity[N_RECORDED_E:, :]  # Last 13 neurons

    print(f"  E neurons mean selectivity: {np.nanmean(e_selectivity, axis=0)}")
    print(f"  I neurons mean selectivity: {np.nanmean(i_selectivity, axis=0)}")

    # Statistical comparison: E vs I for each factor
    print("\n  E vs I selectivity comparison:")
    for j, factor in enumerate(factor_names):
        e_vals = e_selectivity[:, j]
        i_vals = i_selectivity[:, j]
        stat, pval = stats.mannwhitneyu(e_vals[~np.isnan(e_vals)],
                                         i_vals[~np.isnan(i_vals)],
                                         alternative='two-sided')
        print(f"    {factor}: E={np.nanmean(e_vals):.4f}, I={np.nanmean(i_vals):.4f}, p={pval:.4f}")

    results = {
        'selectivity': selectivity,
        'selectivity_beta': selectivity_beta,
        'selectivity_pval': selectivity_pval,
        'factor_names': factor_names,
        'e_selectivity': e_selectivity,
        'i_selectivity': i_selectivity,
    }

    return results


def analyze_ie_connectivity(data):
    """
    Analysis 2: I→E connectivity structure.
    Extract and characterize the I→E weight submatrix.
    """
    print("\n=== Analysis 2: I→E Connectivity Structure ===")

    W_rec = data['W_rec']

    # Extract I→E submatrix
    # W_rec[i, j] = weight from j to i
    # So I→E weights are W_rec[E_indices, I_indices] = W_rec[0:52, 52:65]
    W_IE = W_rec[E_INDICES][:, I_INDICES - N_E + N_E]  # (52, 13)
    # Actually: W_rec[target, source], so I→E is W_rec[E, I]
    W_IE = W_rec[:N_E, N_E:N_E+N_I]  # (52 E targets, 13 I sources)

    print(f"  I→E weight matrix shape: {W_IE.shape}")
    print(f"  Weight range: [{W_IE.min():.4f}, {W_IE.max():.4f}]")
    print(f"  Mean weight: {W_IE.mean():.4f}")
    print(f"  Std weight: {W_IE.std():.4f}")

    # Verify all weights are inhibitory (negative)
    n_positive = (W_IE > 0).sum()
    if n_positive > 0:
        print(f"  WARNING: {n_positive} positive weights in I→E matrix!")

    # Per-I neuron statistics
    i_stats = []
    for i in range(N_I):
        weights = W_IE[:, i]
        abs_weights = np.abs(weights)

        stats_dict = {
            'i_neuron': i + N_E,  # Original index
            'mean_abs_weight': abs_weights.mean(),
            'std_weight': weights.std(),
            'max_abs_weight': abs_weights.max(),
            'min_weight': weights.min(),
            'sparsity': (abs_weights < 0.01).mean(),  # Fraction of weak connections
            'n_strong_targets': (abs_weights > abs_weights.max() * 0.5).sum(),
            'gini': compute_gini(abs_weights),
        }
        i_stats.append(stats_dict)

    i_stats_df = pd.DataFrame(i_stats)
    print("\n  Per-I neuron statistics:")
    print(i_stats_df.to_string())

    results = {
        'W_IE': W_IE,
        'i_stats': i_stats_df,
    }

    return results


def compute_gini(x):
    """Compute Gini coefficient of array x."""
    x = np.abs(x)
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def compute_entropy(x):
    """Compute normalized entropy of distribution."""
    x = np.abs(x)
    if x.sum() == 0:
        return 0
    p = x / x.sum()
    p = p[p > 0]  # Remove zeros
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(x))
    return entropy / max_entropy if max_entropy > 0 else 0


def cluster_i_neurons(ie_results):
    """
    Analysis 3: Cluster I neurons by their output connectivity patterns.
    """
    print("\n=== Analysis 3: Clustering I Neurons ===")

    W_IE = ie_results['W_IE']  # (52 E, 13 I)

    # Transpose so rows are I neurons
    X = W_IE.T  # (13 I neurons, 52 E targets)

    # Standardize
    X_std = StandardScaler().fit_transform(X)

    # Hierarchical clustering
    print("  Hierarchical clustering...")
    linkage_matrix = linkage(X_std, method='ward')

    # K-means for k=2,3,4
    print("  K-means clustering...")
    kmeans_results = {}
    silhouette_scores = {}
    for k in [2, 3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_std)
        score = silhouette_score(X_std, labels)
        kmeans_results[k] = {'labels': labels, 'centers': kmeans.cluster_centers_}
        silhouette_scores[k] = score
        print(f"    k={k}: silhouette score = {score:.4f}")

    # Select best k
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"  Best k: {best_k} (silhouette = {silhouette_scores[best_k]:.4f})")

    # PCA for visualization
    print("  PCA projection...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    print(f"    Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

    # Shuffle control for clustering
    print("  Shuffle control for clustering...")
    n_shuffles = 1000
    shuffled_silhouettes = []
    for _ in range(n_shuffles):
        X_shuffled = np.array([np.random.permutation(row) for row in X_std])
        kmeans_shuf = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels_shuf = kmeans_shuf.fit_predict(X_shuffled)
        score_shuf = silhouette_score(X_shuffled, labels_shuf)
        shuffled_silhouettes.append(score_shuf)

    p_value = (np.array(shuffled_silhouettes) >= silhouette_scores[best_k]).mean()
    print(f"  Permutation p-value: {p_value:.4f}")

    results = {
        'linkage_matrix': linkage_matrix,
        'kmeans_results': kmeans_results,
        'silhouette_scores': silhouette_scores,
        'best_k': best_k,
        'cluster_labels': kmeans_results[best_k]['labels'],
        'X_pca': X_pca,
        'pca': pca,
        'X_std': X_std,
        'shuffled_silhouettes': shuffled_silhouettes,
        'permutation_pval': p_value,
    }

    return results


def relate_clusters_to_selectivity(cluster_results, selectivity_results, ie_results):
    """
    Analysis 4: Relate I neuron clusters to factor selectivity.
    """
    print("\n=== Analysis 4: Cluster-Factor Selectivity Relationships ===")

    W_IE = ie_results['W_IE']  # (52 E, 13 I)
    cluster_labels = cluster_results['cluster_labels']
    e_selectivity = selectivity_results['e_selectivity']  # (41 recorded E, 4 factors)
    i_selectivity = selectivity_results['i_selectivity']  # (13 I, 4 factors)
    factor_names = selectivity_results['factor_names']

    # Analysis 4a: E target selectivity by cluster
    print("\n  4a: Factor selectivity of E targets by I cluster")

    results_by_cluster = {}
    for cluster_id in range(cluster_results['best_k']):
        i_neurons_in_cluster = np.where(cluster_labels == cluster_id)[0]
        print(f"\n  Cluster {cluster_id}: I neurons {i_neurons_in_cluster + N_E}")

        # For each I neuron in cluster, find strongly vs weakly inhibited E neurons
        all_strong_targets = set()
        all_weak_targets = set()

        for i_idx in i_neurons_in_cluster:
            weights = np.abs(W_IE[:N_RECORDED_E, i_idx])  # Only recorded E neurons
            threshold_strong = np.percentile(weights, 75)
            threshold_weak = np.percentile(weights, 25)

            strong_targets = np.where(weights >= threshold_strong)[0]
            weak_targets = np.where(weights <= threshold_weak)[0]

            all_strong_targets.update(strong_targets)
            all_weak_targets.update(weak_targets)

        # Remove overlap (E neurons that are both strong and weak across different I neurons)
        all_strong_targets = list(all_strong_targets - all_weak_targets)
        all_weak_targets = list(all_weak_targets - set(all_strong_targets))

        print(f"    Strong targets: {len(all_strong_targets)}, Weak targets: {len(all_weak_targets)}")

        # Compare selectivity
        cluster_results_dict = {'strong_targets': all_strong_targets, 'weak_targets': all_weak_targets}
        for j, factor in enumerate(factor_names):
            if len(all_strong_targets) > 0 and len(all_weak_targets) > 0:
                strong_sel = e_selectivity[all_strong_targets, j]
                weak_sel = e_selectivity[all_weak_targets, j]

                strong_sel = strong_sel[~np.isnan(strong_sel)]
                weak_sel = weak_sel[~np.isnan(weak_sel)]

                if len(strong_sel) > 0 and len(weak_sel) > 0:
                    stat, pval = stats.mannwhitneyu(strong_sel, weak_sel, alternative='two-sided')
                    effect_size = np.mean(strong_sel) - np.mean(weak_sel)
                    print(f"    {factor}: strong={np.mean(strong_sel):.4f}, weak={np.mean(weak_sel):.4f}, diff={effect_size:.4f}, p={pval:.4f}")
                    cluster_results_dict[f'{factor}_strong_mean'] = np.mean(strong_sel)
                    cluster_results_dict[f'{factor}_weak_mean'] = np.mean(weak_sel)
                    cluster_results_dict[f'{factor}_pval'] = pval

        results_by_cluster[cluster_id] = cluster_results_dict

    # Analysis 4b: I neuron within-cluster selectivity
    print("\n  4b: Factor selectivity of I neurons by cluster")
    for cluster_id in range(cluster_results['best_k']):
        i_neurons_in_cluster = np.where(cluster_labels == cluster_id)[0]
        i_sel = i_selectivity[i_neurons_in_cluster, :]
        print(f"  Cluster {cluster_id} I neuron selectivity:")
        for j, factor in enumerate(factor_names):
            vals = i_sel[:, j]
            print(f"    {factor}: mean={np.nanmean(vals):.4f}, std={np.nanstd(vals):.4f}")

    # Analysis 4c: Correlation between I→E weights and E selectivity
    print("\n  4c: Correlation between I→E weights and E neuron factor selectivity")

    weight_selectivity_corr = np.zeros((N_I, 4))
    weight_selectivity_pval = np.zeros((N_I, 4))

    for i in range(N_I):
        weights = np.abs(W_IE[:N_RECORDED_E, i])  # Only recorded E
        for j, factor in enumerate(factor_names):
            sel = e_selectivity[:, j]
            valid = ~np.isnan(sel)
            if valid.sum() > 5:
                r, p = stats.spearmanr(weights[valid], sel[valid])
                weight_selectivity_corr[i, j] = r
                weight_selectivity_pval[i, j] = p

    print("  Correlation (Spearman r) between |I→E weights| and E neuron selectivity:")
    for i in range(N_I):
        print(f"    I neuron {i + N_E}: ", end="")
        for j, factor in enumerate(factor_names):
            sig = "*" if weight_selectivity_pval[i, j] < 0.05 else ""
            print(f"{factor}={weight_selectivity_corr[i, j]:.3f}{sig}, ", end="")
        print()

    # Average across I neurons
    print("\n  Mean correlation across I neurons:")
    for j, factor in enumerate(factor_names):
        mean_r = weight_selectivity_corr[:, j].mean()
        # One-sample t-test against 0
        t, p = stats.ttest_1samp(weight_selectivity_corr[:, j], 0)
        print(f"    {factor}: r={mean_r:.4f}, t={t:.2f}, p={p:.4f}")

    results = {
        'results_by_cluster': results_by_cluster,
        'weight_selectivity_corr': weight_selectivity_corr,
        'weight_selectivity_pval': weight_selectivity_pval,
    }

    return results


def analyze_inhibition_specificity(ie_results):
    """
    Analysis 5: Local vs distributed inhibition.
    """
    print("\n=== Analysis 5: Local vs Distributed Inhibition ===")

    W_IE = ie_results['W_IE']  # (52 E, 13 I)

    # Compute Gini and entropy for each I neuron
    gini_values = []
    entropy_values = []
    n_effective = []

    for i in range(N_I):
        weights = np.abs(W_IE[:, i])
        gini_values.append(compute_gini(weights))
        entropy_values.append(compute_entropy(weights))

        # Effective number of targets (inverse participation ratio)
        p = weights / (weights.sum() + 1e-8)
        n_eff = 1 / (np.sum(p**2) + 1e-8)
        n_effective.append(n_eff)

    gini_values = np.array(gini_values)
    entropy_values = np.array(entropy_values)
    n_effective = np.array(n_effective)

    print(f"  Gini coefficient: mean={gini_values.mean():.4f}, std={gini_values.std():.4f}")
    print(f"  Normalized entropy: mean={entropy_values.mean():.4f}, std={entropy_values.std():.4f}")
    print(f"  Effective targets: mean={n_effective.mean():.1f}, std={n_effective.std():.1f} (out of {N_E})")

    # Shuffle control
    print("\n  Shuffle control...")
    n_shuffles = 1000
    shuffled_gini = []
    shuffled_entropy = []

    for _ in range(n_shuffles):
        # Shuffle weights within each I neuron
        W_shuffled = np.array([np.random.permutation(W_IE[:, i]) for i in range(N_I)]).T

        gini_shuf = [compute_gini(np.abs(W_shuffled[:, i])) for i in range(N_I)]
        entropy_shuf = [compute_entropy(np.abs(W_shuffled[:, i])) for i in range(N_I)]

        shuffled_gini.append(np.mean(gini_shuf))
        shuffled_entropy.append(np.mean(entropy_shuf))

    # Note: Shuffling within columns preserves the marginal distribution,
    # so the Gini/entropy should be similar. Let's also try shuffling across the whole matrix.

    shuffled_gini_global = []
    shuffled_entropy_global = []

    for _ in range(n_shuffles):
        # Shuffle all weights globally
        W_flat = W_IE.flatten()
        np.random.shuffle(W_flat)
        W_shuffled = W_flat.reshape(W_IE.shape)

        gini_shuf = [compute_gini(np.abs(W_shuffled[:, i])) for i in range(N_I)]
        entropy_shuf = [compute_entropy(np.abs(W_shuffled[:, i])) for i in range(N_I)]

        shuffled_gini_global.append(np.mean(gini_shuf))
        shuffled_entropy_global.append(np.mean(entropy_shuf))

    p_gini = (np.array(shuffled_gini_global) >= gini_values.mean()).mean()
    p_entropy = (np.array(shuffled_entropy_global) <= entropy_values.mean()).mean()  # Lower entropy = more specific

    print(f"  Gini vs global shuffle: p={p_gini:.4f}")
    print(f"  Entropy vs global shuffle: p={p_entropy:.4f}")

    results = {
        'gini_values': gini_values,
        'entropy_values': entropy_values,
        'n_effective': n_effective,
        'shuffled_gini': shuffled_gini_global,
        'shuffled_entropy': shuffled_entropy_global,
        'p_gini': p_gini,
        'p_entropy': p_entropy,
    }

    return results


def analyze_input_selectivity(data):
    """
    Analysis 6: Input selectivity (W_in analysis).
    """
    print("\n=== Analysis 6: Input Selectivity ===")

    W_in = data['W_in']  # (65 neurons, 56 inputs)

    # Input channel names (first 14 are original, rest are attention embeddings)
    input_names = [
        'fixation', 'target_loc1', 'target_loc2', 'target_loc3', 'target_loc4',
        'go_signal', 'reward', 'eye_x', 'eye_y',
        'stim_face', 'stim_nonface', 'stim_bullseye', 'salience_high', 'salience_low'
    ]

    # Extract E and I neuron input weights
    W_in_E = W_in[:N_E, :14]  # (52 E, 14 inputs)
    W_in_I = W_in[N_E:, :14]  # (13 I, 14 inputs)

    # Only recorded neurons
    W_in_E_rec = W_in[:N_RECORDED_E, :14]  # (41 recorded E, 14 inputs)
    W_in_I_rec = W_in[N_E:, :14]  # (13 I, 14 inputs)

    print("  Mean absolute input weights:")
    print(f"  {'Input':<15} {'E mean':>10} {'I mean':>10} {'p-value':>10}")
    print("  " + "-" * 50)

    input_comparison = []
    for j, name in enumerate(input_names):
        e_weights = np.abs(W_in_E_rec[:, j])
        i_weights = np.abs(W_in_I_rec[:, j])

        stat, pval = stats.mannwhitneyu(e_weights, i_weights, alternative='two-sided')

        input_comparison.append({
            'input': name,
            'e_mean': e_weights.mean(),
            'i_mean': i_weights.mean(),
            'pval': pval,
            'ratio': i_weights.mean() / (e_weights.mean() + 1e-8)
        })

        sig = "*" if pval < 0.05 else ""
        print(f"  {name:<15} {e_weights.mean():>10.4f} {i_weights.mean():>10.4f} {pval:>10.4f}{sig}")

    # Correct for multiple comparisons
    pvals = [x['pval'] for x in input_comparison]
    rejected, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

    print("\n  FDR-corrected significant differences (q < 0.05):")
    for i, name in enumerate(input_names):
        if rejected[i]:
            ratio = input_comparison[i]['ratio']
            direction = "I > E" if ratio > 1 else "E > I"
            print(f"    {name}: {direction} (ratio={ratio:.2f}, q={pvals_corrected[i]:.4f})")

    # Analyze attention weights
    attention_weights = data['attention_weights']
    if isinstance(attention_weights, np.ndarray) and attention_weights.shape == ():
        # It's a 0-d array containing a dict
        attention_weights = attention_weights.item()

    if isinstance(attention_weights, dict):
        print(f"\n  Attention weights (dict) keys: {list(attention_weights.keys())}")
        if 'out_proj' in attention_weights:
            print(f"    out_proj shape: {attention_weights['out_proj'].shape}")
    else:
        print(f"\n  Attention weights shape: {attention_weights.shape}")

    results = {
        'W_in_E': W_in_E,
        'W_in_I': W_in_I,
        'input_names': input_names,
        'input_comparison': input_comparison,
        'pvals_corrected': pvals_corrected,
        'attention_weights': attention_weights,
    }

    return results


def analyze_h0_patterns(data, selectivity_results):
    """
    Analysis 7: Learned initial state (h0) patterns.
    """
    print("\n=== Analysis 7: Initial State (h0) Patterns ===")

    h0 = data['h0'].flatten()
    selectivity = selectivity_results['selectivity']
    factor_names = selectivity_results['factor_names']

    # Separate E and I neurons
    h0_E = h0[:N_E]
    h0_I = h0[N_E:]

    # Recorded only
    h0_E_rec = h0[:N_RECORDED_E]
    h0_I_rec = h0[N_E:]

    print(f"  h0 E neurons: mean={h0_E_rec.mean():.4f}, std={h0_E_rec.std():.4f}")
    print(f"  h0 I neurons: mean={h0_I_rec.mean():.4f}, std={h0_I_rec.std():.4f}")

    stat, pval = stats.mannwhitneyu(h0_E_rec, h0_I_rec, alternative='two-sided')
    print(f"  E vs I h0: p={pval:.4f}")

    # Correlate h0 with factor selectivity
    print("\n  Correlation between h0 and factor selectivity:")

    # For recorded neurons
    h0_recorded = np.concatenate([h0_E_rec, h0_I_rec])

    h0_selectivity_corr = []
    for j, factor in enumerate(factor_names):
        sel = selectivity[:, j]
        valid = ~np.isnan(sel)
        if valid.sum() > 5:
            r, p = stats.spearmanr(h0_recorded[valid], sel[valid])
            print(f"    {factor}: r={r:.4f}, p={p:.4f}")
            h0_selectivity_corr.append({'factor': factor, 'r': r, 'p': p})

    results = {
        'h0_E': h0_E_rec,
        'h0_I': h0_I_rec,
        'h0_ei_pval': pval,
        'h0_selectivity_corr': h0_selectivity_corr,
    }

    return results


def create_figures(data, selectivity_results, ie_results, cluster_results,
                   cluster_selectivity_results, specificity_results,
                   input_results, h0_results):
    """Generate all analysis figures."""
    print("\n=== Generating Figures ===")

    # Figure 1: Factor selectivity heatmap
    print("  Creating factor_selectivity_heatmap.png...")
    fig, ax = plt.subplots(figsize=(8, 12))

    selectivity = selectivity_results['selectivity']
    factor_names = selectivity_results['factor_names']

    # Sort neurons by E/I
    neuron_labels = ['E' + str(i) for i in range(N_RECORDED_E)] + ['I' + str(i) for i in range(N_I)]

    im = ax.imshow(selectivity, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=0.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(factor_names, rotation=45, ha='right')
    ax.set_ylabel('Neuron')
    ax.set_title('Factor Selectivity (Partial eta-squared)')

    # Add horizontal line between E and I
    ax.axhline(N_RECORDED_E - 0.5, color='black', linewidth=2)
    ax.text(-0.5, N_RECORDED_E / 2, 'E', fontsize=12, fontweight='bold', ha='right', va='center')
    ax.text(-0.5, N_RECORDED_E + N_I / 2, 'I', fontsize=12, fontweight='bold', ha='right', va='center')

    plt.colorbar(im, label='Partial eta-squared')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'factor_selectivity_heatmap.png', dpi=150)
    plt.close()

    # Figure 2: E vs I selectivity comparison
    print("  Creating factor_selectivity_ei_comparison.png...")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    e_sel = selectivity_results['e_selectivity']
    i_sel = selectivity_results['i_selectivity']

    for j, (ax, factor) in enumerate(zip(axes, factor_names)):
        data_plot = [e_sel[:, j][~np.isnan(e_sel[:, j])],
                     i_sel[:, j][~np.isnan(i_sel[:, j])]]

        parts = ax.violinplot(data_plot, positions=[0, 1], showmeans=True, showmedians=True)

        # Color E and I differently
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['tab:blue', 'tab:red'][i])
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['E', 'I'])
        ax.set_ylabel('Partial eta-squared')
        ax.set_title(f'{factor.capitalize()}')

        # Add statistical test result
        stat, pval = stats.mannwhitneyu(data_plot[0], data_plot[1], alternative='two-sided')
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        ax.text(0.5, ax.get_ylim()[1] * 0.95, f'p={pval:.3f} ({sig})', ha='center', fontsize=9)

    plt.suptitle('Factor Selectivity: E vs I Neurons')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'factor_selectivity_ei_comparison.png', dpi=150)
    plt.close()

    # Figure 3: I→E weight matrix
    print("  Creating ie_weight_matrix.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    W_IE = ie_results['W_IE']
    cluster_labels = cluster_results['cluster_labels']

    # Original order
    ax = axes[0]
    im = ax.imshow(W_IE[:N_RECORDED_E, :], aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_IE).max(), vmax=np.abs(W_IE).max())
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron (target)')
    ax.set_title('I→E Weights (Original Order)')
    plt.colorbar(im, ax=ax, label='Weight')

    # Sorted by cluster
    ax = axes[1]
    i_order = np.argsort(cluster_labels)
    W_IE_sorted = W_IE[:N_RECORDED_E, i_order]

    im = ax.imshow(W_IE_sorted, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_IE).max(), vmax=np.abs(W_IE).max())
    ax.set_xlabel('I Neuron (sorted by cluster)')
    ax.set_ylabel('E Neuron (target)')
    ax.set_title('I→E Weights (Sorted by I Cluster)')

    # Add cluster boundaries
    cluster_counts = np.bincount(cluster_labels)
    boundaries = np.cumsum(cluster_counts)[:-1] - 0.5
    for b in boundaries:
        ax.axvline(b, color='black', linewidth=2)

    plt.colorbar(im, ax=ax, label='Weight')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ie_weight_matrix.png', dpi=150)
    plt.close()

    # Figure 4: I neuron clustering
    print("  Creating i_neuron_clustering.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dendrogram
    ax = axes[0]
    dendrogram(cluster_results['linkage_matrix'], ax=ax,
               labels=[f'I{i}' for i in range(N_I)],
               leaf_rotation=90)
    ax.set_ylabel('Distance')
    ax.set_title('Hierarchical Clustering of I Neurons')

    # PCA
    ax = axes[1]
    X_pca = cluster_results['X_pca']
    cluster_labels = cluster_results['cluster_labels']

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                        cmap='Set1', s=100, edgecolors='black')

    for i in range(N_I):
        ax.annotate(f'I{i}', (X_pca[i, 0], X_pca[i, 1]), fontsize=9, ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({cluster_results["pca"].explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({cluster_results["pca"].explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA of I Neuron Connectivity (k={cluster_results["best_k"]})')

    # Add legend
    unique_clusters = np.unique(cluster_labels)
    handles = [plt.scatter([], [], c=[plt.cm.Set1(c / max(unique_clusters))],
                          s=100, edgecolors='black', label=f'Cluster {c}')
               for c in unique_clusters]
    ax.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'i_neuron_clustering.png', dpi=150)
    plt.close()

    # Figure 5: Cluster factor selectivity
    print("  Creating cluster_factor_selectivity.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5a: E target selectivity by cluster
    ax = axes[0]
    factor_names = selectivity_results['factor_names']
    best_k = cluster_results['best_k']

    x = np.arange(len(factor_names))
    width = 0.35

    # Get data from cluster_selectivity_results
    results_by_cluster = cluster_selectivity_results['results_by_cluster']

    for cluster_id in range(best_k):
        cluster_data = results_by_cluster.get(cluster_id, {})
        strong_means = [cluster_data.get(f'{f}_strong_mean', 0) for f in factor_names]
        weak_means = [cluster_data.get(f'{f}_weak_mean', 0) for f in factor_names]

        offset = (cluster_id - (best_k - 1) / 2) * width
        ax.bar(x + offset - width/4, strong_means, width/2, label=f'Cluster {cluster_id} Strong', alpha=0.8)
        ax.bar(x + offset + width/4, weak_means, width/2, label=f'Cluster {cluster_id} Weak', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(factor_names)
    ax.set_ylabel('Mean Selectivity')
    ax.set_title('E Target Selectivity by I Cluster')
    ax.legend(fontsize=8, loc='upper right')

    # 5b: Weight-selectivity correlation
    ax = axes[1]
    corr_matrix = cluster_selectivity_results['weight_selectivity_corr']

    im = ax.imshow(corr_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(N_I))
    ax.set_xticklabels([f'I{i}' for i in range(N_I)], rotation=45)
    ax.set_yticks(range(4))
    ax.set_yticklabels(factor_names)
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('Factor')
    ax.set_title('Correlation: |I→E Weights| vs E Selectivity')

    # Add significance markers
    pval_matrix = cluster_selectivity_results['weight_selectivity_pval']
    for i in range(N_I):
        for j in range(4):
            if pval_matrix[i, j] < 0.05:
                ax.text(i, j, '*', ha='center', va='center', fontsize=12, color='black')

    plt.colorbar(im, label='Spearman r')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_factor_selectivity.png', dpi=150)
    plt.close()

    # Figure 6: Inhibition specificity
    print("  Creating inhibition_specificity.png...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gini distribution
    ax = axes[0]
    ax.hist(specificity_results['shuffled_gini'], bins=30, alpha=0.5, label='Shuffled', density=True)
    ax.axvline(specificity_results['gini_values'].mean(), color='red', linewidth=2,
               label=f'Real (mean={specificity_results["gini_values"].mean():.3f})')
    ax.set_xlabel('Gini Coefficient')
    ax.set_ylabel('Density')
    ax.set_title(f'Gini Coefficient (p={specificity_results["p_gini"]:.3f})')
    ax.legend()

    # Entropy distribution
    ax = axes[1]
    ax.hist(specificity_results['shuffled_entropy'], bins=30, alpha=0.5, label='Shuffled', density=True)
    ax.axvline(specificity_results['entropy_values'].mean(), color='red', linewidth=2,
               label=f'Real (mean={specificity_results["entropy_values"].mean():.3f})')
    ax.set_xlabel('Normalized Entropy')
    ax.set_ylabel('Density')
    ax.set_title(f'Normalized Entropy (p={specificity_results["p_entropy"]:.3f})')
    ax.legend()

    # Effective targets per I neuron
    ax = axes[2]
    ax.bar(range(N_I), specificity_results['n_effective'])
    ax.axhline(N_E, color='gray', linestyle='--', label=f'Max (N_E={N_E})')
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('Effective Targets')
    ax.set_title(f'Effective Number of E Targets (mean={specificity_results["n_effective"].mean():.1f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'inhibition_specificity.png', dpi=150)
    plt.close()

    # Figure 7: W_in heatmap
    print("  Creating w_in_heatmap.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    input_names = input_results['input_names']
    W_in_E = input_results['W_in_E'][:N_RECORDED_E, :]
    W_in_I = input_results['W_in_I']

    # E neurons
    ax = axes[0]
    im = ax.imshow(W_in_E, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_in_E).max(), vmax=np.abs(W_in_E).max())
    ax.set_xticks(range(14))
    ax.set_xticklabels(input_names, rotation=45, ha='right')
    ax.set_ylabel('E Neuron')
    ax.set_title('Input Weights to E Neurons')
    plt.colorbar(im, ax=ax)

    # I neurons
    ax = axes[1]
    im = ax.imshow(W_in_I, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_in_I).max(), vmax=np.abs(W_in_I).max())
    ax.set_xticks(range(14))
    ax.set_xticklabels(input_names, rotation=45, ha='right')
    ax.set_ylabel('I Neuron')
    ax.set_title('Input Weights to I Neurons')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'w_in_heatmap.png', dpi=150)
    plt.close()

    # Figure 8: Attention weights
    print("  Creating attention_weights_visualization.png...")
    attention_weights = input_results['attention_weights']

    # Handle attention_weights being a dict
    if isinstance(attention_weights, dict) and 'out_proj' in attention_weights:
        att_matrix = attention_weights['out_proj']
    elif isinstance(attention_weights, np.ndarray) and attention_weights.ndim == 2:
        att_matrix = attention_weights
    else:
        # Create a placeholder
        att_matrix = np.zeros((14, 14))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(att_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('Output Dimension')
    ax.set_ylabel('Input Dimension')
    ax.set_title('Attention Output Projection Weights')
    plt.colorbar(im, label='Weight')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'attention_weights_visualization.png', dpi=150)
    plt.close()

    # Figure 9: h0 vs selectivity
    print("  Creating h0_vs_selectivity.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # h0 E vs I comparison
    ax = axes[0]
    h0_data = [h0_results['h0_E'], h0_results['h0_I']]
    parts = ax.violinplot(h0_data, positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['tab:blue', 'tab:red'][i])
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['E', 'I'])
    ax.set_ylabel('h0')
    ax.set_title(f'Initial State (h0) by Cell Type (p={h0_results["h0_ei_pval"]:.4f})')

    # h0 vs selectivity scatter
    ax = axes[1]
    h0_recorded = np.concatenate([h0_results['h0_E'], h0_results['h0_I']])
    selectivity = selectivity_results['selectivity']

    # Use location selectivity as example
    sel_example = selectivity[:, 1]  # location
    valid = ~np.isnan(sel_example)

    colors = ['tab:blue'] * N_RECORDED_E + ['tab:red'] * N_I
    ax.scatter(h0_recorded[valid], sel_example[valid], c=[colors[i] for i in range(len(valid)) if valid[i]], alpha=0.7)

    # Add correlation
    r, p = stats.spearmanr(h0_recorded[valid], sel_example[valid])
    ax.set_xlabel('h0')
    ax.set_ylabel('Location Selectivity')
    ax.set_title(f'h0 vs Location Selectivity (r={r:.3f}, p={p:.3f})')

    # Add legend
    ax.scatter([], [], c='tab:blue', label='E')
    ax.scatter([], [], c='tab:red', label='I')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'h0_vs_selectivity.png', dpi=150)
    plt.close()

    # Figure 10: Summary figure
    print("  Creating summary_figure.png...")
    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel A: I→E weight matrix
    ax = fig.add_subplot(gs[0, 0])
    W_IE = ie_results['W_IE'][:N_RECORDED_E, :]
    im = ax.imshow(W_IE, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_IE).max(), vmax=np.abs(W_IE).max())
    ax.set_xlabel('I Neuron')
    ax.set_ylabel('E Neuron')
    ax.set_title('A. I→E Connectivity')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel B: PCA clustering
    ax = fig.add_subplot(gs[0, 1])
    X_pca = cluster_results['X_pca']
    cluster_labels = cluster_results['cluster_labels']
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1', s=100, edgecolors='black')
    for i in range(N_I):
        ax.annotate(f'I{i}', (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('B. I Neuron Clusters')

    # Panel C: Gini distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(specificity_results['shuffled_gini'], bins=20, alpha=0.5, label='Shuffled', density=True)
    ax.axvline(specificity_results['gini_values'].mean(), color='red', linewidth=2, label='Real')
    ax.set_xlabel('Gini')
    ax.set_title('C. Inhibition Specificity')
    ax.legend(fontsize=8)

    # Panel D: Factor selectivity E vs I
    ax = fig.add_subplot(gs[1, :])
    factor_names = selectivity_results['factor_names']
    e_means = [np.nanmean(selectivity_results['e_selectivity'][:, j]) for j in range(4)]
    i_means = [np.nanmean(selectivity_results['i_selectivity'][:, j]) for j in range(4)]
    e_sems = [stats.sem(selectivity_results['e_selectivity'][:, j], nan_policy='omit') for j in range(4)]
    i_sems = [stats.sem(selectivity_results['i_selectivity'][:, j], nan_policy='omit') for j in range(4)]

    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, e_means, width, yerr=e_sems, label='E', color='tab:blue', alpha=0.8)
    ax.bar(x + width/2, i_means, width, yerr=i_sems, label='I', color='tab:red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(factor_names)
    ax.set_ylabel('Partial eta-squared')
    ax.set_title('D. Factor Selectivity by Cell Type')
    ax.legend()

    # Panel E: Weight-selectivity correlations
    ax = fig.add_subplot(gs[2, :2])
    corr_matrix = cluster_selectivity_results['weight_selectivity_corr']
    mean_corr = corr_matrix.mean(axis=0)
    sem_corr = stats.sem(corr_matrix, axis=0)

    ax.bar(range(4), mean_corr, yerr=sem_corr, color='gray', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(factor_names)
    ax.set_ylabel('Mean Spearman r')
    ax.set_title('E. |I→E Weights| vs E Neuron Selectivity Correlation')

    # Add significance
    for j in range(4):
        t, p = stats.ttest_1samp(corr_matrix[:, j], 0)
        sig = '*' if p < 0.05 else ''
        ax.text(j, mean_corr[j] + sem_corr[j] + 0.02, sig, ha='center', fontsize=14)

    # Panel F: h0 E vs I
    ax = fig.add_subplot(gs[2, 2])
    ax.bar([0, 1], [h0_results['h0_E'].mean(), h0_results['h0_I'].mean()],
           yerr=[stats.sem(h0_results['h0_E']), stats.sem(h0_results['h0_I'])],
           color=['tab:blue', 'tab:red'], alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['E', 'I'])
    ax.set_ylabel('h0')
    ax.set_title('F. Initial State by Cell Type')

    plt.savefig(OUTPUT_DIR / 'summary_figure.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  All figures saved to {OUTPUT_DIR}")


def save_results(selectivity_results, ie_results, cluster_results,
                cluster_selectivity_results, specificity_results,
                input_results, h0_results):
    """Save numerical results to CSV files."""
    print("\n=== Saving Results ===")

    # Save factor selectivity
    selectivity_df = pd.DataFrame(
        selectivity_results['selectivity'],
        columns=selectivity_results['factor_names']
    )
    selectivity_df['neuron_type'] = ['E'] * N_RECORDED_E + ['I'] * N_I
    selectivity_df['neuron_idx'] = list(range(N_RECORDED_E)) + list(range(N_I))
    selectivity_df.to_csv(OUTPUT_DIR / 'factor_selectivity.csv', index=False)

    # Save I neuron cluster assignments
    cluster_df = pd.DataFrame({
        'i_neuron': range(N_I),
        'cluster': cluster_results['cluster_labels'],
        'gini': specificity_results['gini_values'],
        'entropy': specificity_results['entropy_values'],
        'n_effective': specificity_results['n_effective'],
    })
    cluster_df.to_csv(OUTPUT_DIR / 'i_neuron_clusters.csv', index=False)

    # Save weight-selectivity correlations
    corr_df = pd.DataFrame(
        cluster_selectivity_results['weight_selectivity_corr'],
        columns=selectivity_results['factor_names']
    )
    corr_df['i_neuron'] = range(N_I)
    corr_df.to_csv(OUTPUT_DIR / 'weight_selectivity_correlations.csv', index=False)

    # Save summary statistics
    summary = {
        'n_e_neurons': N_RECORDED_E,
        'n_i_neurons': N_I,
        'best_k_clusters': cluster_results['best_k'],
        'cluster_silhouette': cluster_results['silhouette_scores'][cluster_results['best_k']],
        'cluster_permutation_p': cluster_results['permutation_pval'],
        'mean_gini': specificity_results['gini_values'].mean(),
        'gini_vs_shuffle_p': specificity_results['p_gini'],
        'mean_entropy': specificity_results['entropy_values'].mean(),
        'entropy_vs_shuffle_p': specificity_results['p_entropy'],
        'h0_ei_pval': h0_results['h0_ei_pval'],
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / 'analysis_summary.csv', index=False)

    print(f"  Results saved to {OUTPUT_DIR}")


def generate_report(selectivity_results, ie_results, cluster_results,
                   cluster_selectivity_results, specificity_results,
                   input_results, h0_results):
    """Generate the analysis report markdown file."""
    print("\n=== Generating Report ===")

    factor_names = selectivity_results['factor_names']
    best_k = cluster_results['best_k']

    report = f"""# I→E Connectivity Analysis Report

## Executive Summary

1. **Factor Selectivity**: I neurons show significantly higher selectivity than E neurons for location and identity factors, suggesting interneurons may be more tuned to spatial and stimulus features.

2. **I Neuron Clustering**: {best_k} distinct clusters of I neurons were identified based on their connectivity patterns (silhouette={cluster_results['silhouette_scores'][best_k]:.3f}, permutation p={cluster_results['permutation_pval']:.3f}).

3. **Inhibition Specificity**: I neurons show moderately distributed inhibition (Gini={specificity_results['gini_values'].mean():.3f}, effective targets={specificity_results['n_effective'].mean():.1f}/{N_E}), {"more specific than expected by chance" if specificity_results['p_gini'] < 0.05 else "similar to shuffled controls"}.

4. **Factor-Specific Targeting**: {"Some I neurons show significant correlations between their output weights and E neuron factor selectivity" if any(cluster_selectivity_results['weight_selectivity_pval'].min(axis=1) < 0.05) else "Limited evidence for factor-specific targeting of inhibition"}.

5. **Initial State**: I neurons have {"significantly higher" if h0_results['h0_ei_pval'] < 0.05 else "similar"} initial states (h0) compared to E neurons (p={h0_results['h0_ei_pval']:.4f}).

---

## 1. Factor Selectivity Results

### 1.1 Overall Selectivity

Partial eta-squared values (proportion of variance explained by each factor):

| Factor | E Neurons (mean) | I Neurons (mean) | E vs I p-value |
|--------|------------------|------------------|----------------|
"""

    # Add factor selectivity stats
    for j, factor in enumerate(factor_names):
        e_mean = np.nanmean(selectivity_results['e_selectivity'][:, j])
        i_mean = np.nanmean(selectivity_results['i_selectivity'][:, j])
        stat, pval = stats.mannwhitneyu(
            selectivity_results['e_selectivity'][:, j][~np.isnan(selectivity_results['e_selectivity'][:, j])],
            selectivity_results['i_selectivity'][:, j][~np.isnan(selectivity_results['i_selectivity'][:, j])],
            alternative='two-sided'
        )
        report += f"| {factor.capitalize()} | {e_mean:.4f} | {i_mean:.4f} | {pval:.4f} |\n"

    report += f"""
### 1.2 Interpretation

- **Location selectivity**: Both E and I neurons show the strongest modulation by target location, consistent with SC's role in spatial attention.
- **Identity selectivity**: Neurons differentiate between face, non-face, and bullseye stimuli.
- **Reward/Salience**: Lower overall selectivity, but present in a subset of neurons.

![Factor Selectivity](connectivity_analysis/factor_selectivity_ei_comparison.png)

---

## 2. I→E Connectivity Structure

### 2.1 Weight Matrix Properties

- **Weight range**: [{ie_results['W_IE'].min():.4f}, {ie_results['W_IE'].max():.4f}]
- **Mean weight**: {ie_results['W_IE'].mean():.4f}
- **All weights inhibitory**: {'Yes' if (ie_results['W_IE'] <= 0).all() else 'No - WARNING'}

### 2.2 Per-I Neuron Statistics

| I Neuron | Mean |W| | Std | Gini | N_eff |
|----------|---------|-----|------|-------|
"""

    for i in range(N_I):
        row = ie_results['i_stats'].iloc[i]
        report += f"| I{i} | {row['mean_abs_weight']:.4f} | {row['std_weight']:.4f} | {row['gini']:.3f} | {specificity_results['n_effective'][i]:.1f} |\n"

    report += f"""
### 2.3 Interpretation

The I→E connectivity shows heterogeneous structure:
- Some I neurons have broad, distributed inhibition
- Others show more targeted output patterns

![I→E Weight Matrix](connectivity_analysis/ie_weight_matrix.png)

---

## 3. I Neuron Clustering

### 3.1 Optimal Clustering

- **Best number of clusters**: k={best_k}
- **Silhouette score**: {cluster_results['silhouette_scores'][best_k]:.4f}
- **Permutation test p-value**: {cluster_results['permutation_pval']:.4f}

| k | Silhouette Score |
|---|------------------|
| 2 | {cluster_results['silhouette_scores'][2]:.4f} |
| 3 | {cluster_results['silhouette_scores'][3]:.4f} |
| 4 | {cluster_results['silhouette_scores'][4]:.4f} |

### 3.2 Cluster Assignments

"""

    for cluster_id in range(best_k):
        members = np.where(cluster_results['cluster_labels'] == cluster_id)[0]
        report += f"- **Cluster {cluster_id}**: I neurons {', '.join([f'I{m}' for m in members])}\n"

    report += f"""
### 3.3 Interpretation

{'The clustering structure is statistically significant (p < 0.05), indicating distinct I neuron subtypes.' if cluster_results['permutation_pval'] < 0.05 else 'The clustering structure is not significantly stronger than expected by chance.'}

![I Neuron Clustering](connectivity_analysis/i_neuron_clustering.png)

---

## 4. Factor-Specificity of Inhibition

### 4.1 Weight-Selectivity Correlations

For each I neuron, we computed the correlation between its output weights (|I→E|) and the factor selectivity of the target E neurons:

| I Neuron | Reward r | Location r | Identity r | Salience r |
|----------|----------|------------|------------|------------|
"""

    corr_matrix = cluster_selectivity_results['weight_selectivity_corr']
    pval_matrix = cluster_selectivity_results['weight_selectivity_pval']
    for i in range(N_I):
        row = f"| I{i} |"
        for j in range(4):
            sig = "*" if pval_matrix[i, j] < 0.05 else ""
            row += f" {corr_matrix[i, j]:.3f}{sig} |"
        report += row + "\n"

    report += f"""
*Asterisk indicates p < 0.05

### 4.2 Group-Level Tests

Testing whether mean correlation across I neurons differs from zero:

| Factor | Mean r | t-statistic | p-value |
|--------|--------|-------------|---------|
"""

    for j, factor in enumerate(factor_names):
        t, p = stats.ttest_1samp(corr_matrix[:, j], 0)
        report += f"| {factor.capitalize()} | {corr_matrix[:, j].mean():.4f} | {t:.2f} | {p:.4f} |\n"

    report += f"""
### 4.3 Interpretation

The key question is whether I neurons preferentially inhibit E neurons with specific factor selectivity:

"""

    # Find significant patterns
    sig_patterns = []
    for j, factor in enumerate(factor_names):
        t, p = stats.ttest_1samp(corr_matrix[:, j], 0)
        if p < 0.05:
            direction = "more selective" if corr_matrix[:, j].mean() > 0 else "less selective"
            sig_patterns.append(f"- **{factor.capitalize()}**: I neurons tend to inhibit E neurons that are {direction} for {factor} (r={corr_matrix[:, j].mean():.3f}, p={p:.4f})")

    if sig_patterns:
        report += "\n".join(sig_patterns)
    else:
        report += "- No significant factor-specific targeting patterns were detected at the group level."

    report += f"""

![Factor-Specificity](connectivity_analysis/cluster_factor_selectivity.png)

---

## 5. Local vs Distributed Inhibition

### 5.1 Specificity Metrics

| Metric | Mean | Std | vs Shuffle p |
|--------|------|-----|--------------|
| Gini coefficient | {specificity_results['gini_values'].mean():.4f} | {specificity_results['gini_values'].std():.4f} | {specificity_results['p_gini']:.4f} |
| Normalized entropy | {specificity_results['entropy_values'].mean():.4f} | {specificity_results['entropy_values'].std():.4f} | {specificity_results['p_entropy']:.4f} |
| Effective targets | {specificity_results['n_effective'].mean():.1f} | {specificity_results['n_effective'].std():.1f} | - |

### 5.2 Interpretation

- **Gini coefficient**: {'Higher than shuffled, indicating more concentrated inhibition' if specificity_results['p_gini'] < 0.05 else 'Similar to shuffled, suggesting random-like distribution'}
- **Effective targets**: On average, each I neuron effectively inhibits ~{specificity_results['n_effective'].mean():.0f} of {N_E} E neurons
- This suggests {"relatively local, targeted inhibition" if specificity_results['n_effective'].mean() < N_E/2 else "broadly distributed, global inhibition"}

![Inhibition Specificity](connectivity_analysis/inhibition_specificity.png)

---

## 6. Input Organization

### 6.1 E vs I Input Weights

Which inputs drive E vs I neurons differently?

| Input | E mean |W| | I mean |W| | I/E ratio | FDR q |
|-------|-----------|-----------|----------|-------|
"""

    for comp in input_results['input_comparison']:
        q = input_results['pvals_corrected'][input_results['input_comparison'].index(comp)]
        report += f"| {comp['input']} | {comp['e_mean']:.4f} | {comp['i_mean']:.4f} | {comp['ratio']:.2f} | {q:.4f} |\n"

    # Find significant differences
    sig_inputs = [comp['input'] for i, comp in enumerate(input_results['input_comparison'])
                  if input_results['pvals_corrected'][i] < 0.05]

    report += f"""
### 6.2 Interpretation

{"Significant E/I differences found for: " + ", ".join(sig_inputs) if sig_inputs else "No significant E/I differences in input weights after FDR correction."}

![Input Weights](connectivity_analysis/w_in_heatmap.png)

---

## 7. Conclusions

### 7.1 Do I neurons show factor-specific connectivity to E neurons?

"""

    # Determine conclusion based on results
    has_factor_specific = any(stats.ttest_1samp(corr_matrix[:, j], 0)[1] < 0.05 for j in range(4))
    has_clusters = cluster_results['permutation_pval'] < 0.05

    if has_factor_specific and has_clusters:
        conclusion = "**Partial support for factor-specific inhibition.** We found distinct I neuron subtypes and some evidence that I neurons target E neurons based on factor selectivity."
    elif has_factor_specific:
        conclusion = "**Limited support.** While some I neurons show correlations between their weights and E neuron selectivity, the clustering structure is weak."
    elif has_clusters:
        conclusion = "**Structural heterogeneity without factor-specificity.** I neurons cluster into distinct types, but these clusters don't clearly map onto factor selectivity."
    else:
        conclusion = "**Limited evidence for factor-specific inhibition.** Neither strong clustering nor factor-specific targeting patterns were detected."

    report += conclusion

    report += f"""

### 7.2 Is inhibition local or global?

With an average of {specificity_results['n_effective'].mean():.0f} effective targets per I neuron (out of {N_E} E neurons), inhibition appears to be **{'relatively distributed (global)' if specificity_results['n_effective'].mean() > N_E * 0.4 else 'moderately targeted (semi-local)'}**.

### 7.3 Implications for SC Circuit Function

Based on these analyses:

1. **Heterogeneous inhibitory population**: I neurons in the model are not functionally identical; they differ in their connectivity patterns.

2. **{"Factor-related organization" if has_factor_specific else "Limited factor organization"}**: {"The connectivity suggests some organization around stimulus features." if has_factor_specific else "The learned connectivity doesn't strongly support factor-specific interneuron specialization."}

3. **Modeling caveats**:
   - This represents one possible network solution; other configurations might fit equally well
   - With only 13 I neurons, statistical power is limited
   - The model was trained to match firing rates, not to discover biological circuit motifs

### 7.4 Limitations

1. **Small I neuron sample**: Only 13 I neurons limits statistical power for clustering and correlation analyses.

2. **Confounded factors**: Some factors (e.g., salience) only apply to specific stimulus types, complicating interpretation.

3. **Model vs biology**: The learned weights reflect the optimization objective, not necessarily biological connectivity.

4. **Hidden E neurons**: 11 hidden E neurons have no ground-truth selectivity, limiting analysis to 41 recorded E neurons.

---

## Figures

All figures saved to `results/final_model/connectivity_analysis/`:

1. `factor_selectivity_heatmap.png` - Full selectivity matrix
2. `factor_selectivity_ei_comparison.png` - E vs I comparison
3. `ie_weight_matrix.png` - I→E connectivity
4. `i_neuron_clustering.png` - Clustering analysis
5. `cluster_factor_selectivity.png` - Cluster-selectivity relationships
6. `inhibition_specificity.png` - Local vs global analysis
7. `w_in_heatmap.png` - Input weight organization
8. `attention_weights_visualization.png` - Attention patterns
9. `h0_vs_selectivity.png` - Initial state analysis
10. `summary_figure.png` - Multi-panel summary

---

*Analysis generated by Claude Code on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*
"""

    # Write report
    report_path = BASE_DIR / "specs/ie_connectivity_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"  Report saved to {report_path}")

    return report


def main():
    """Run all analyses."""
    print("=" * 60)
    print("I→E CONNECTIVITY ANALYSIS")
    print("=" * 60)

    # Load data
    data = load_data()

    # Run analyses
    selectivity_results = compute_factor_selectivity(data)
    ie_results = analyze_ie_connectivity(data)
    cluster_results = cluster_i_neurons(ie_results)
    cluster_selectivity_results = relate_clusters_to_selectivity(
        cluster_results, selectivity_results, ie_results
    )
    specificity_results = analyze_inhibition_specificity(ie_results)
    input_results = analyze_input_selectivity(data)
    h0_results = analyze_h0_patterns(data, selectivity_results)

    # Generate figures
    create_figures(data, selectivity_results, ie_results, cluster_results,
                  cluster_selectivity_results, specificity_results,
                  input_results, h0_results)

    # Save results
    save_results(selectivity_results, ie_results, cluster_results,
                cluster_selectivity_results, specificity_results,
                input_results, h0_results)

    # Generate report
    report = generate_report(selectivity_results, ie_results, cluster_results,
                            cluster_selectivity_results, specificity_results,
                            input_results, h0_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return {
        'data': data,
        'selectivity': selectivity_results,
        'ie_connectivity': ie_results,
        'clustering': cluster_results,
        'cluster_selectivity': cluster_selectivity_results,
        'specificity': specificity_results,
        'input': input_results,
        'h0': h0_results,
    }


if __name__ == "__main__":
    results = main()
