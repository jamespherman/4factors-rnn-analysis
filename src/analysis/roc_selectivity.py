"""
ROC-based selectivity analysis matching neuronal analysis pipeline.

Uses ROC-AUC (area under ROC curve) computed via Mann-Whitney U statistic,
with permutation-based significance testing and target-onset aligned time windows.

Matches the approach in 4factors-analysis-pipeline/code/utils/arrayROC.m
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any


@dataclass
class ROCSelectivityResult:
    """Container for ROC-based selectivity results."""
    factor_name: str
    auc: np.ndarray              # [n_neurons] AUC values (0.5 = chance)
    p_values: np.ndarray         # [n_neurons] permutation p-values
    ci_lower: np.ndarray         # [n_neurons] bootstrap CI lower bound
    ci_upper: np.ndarray         # [n_neurons] bootstrap CI upper bound
    n_trials_cond1: np.ndarray   # [n_neurons] trial counts
    n_trials_cond2: np.ndarray
    significant: np.ndarray      # [n_neurons] bool - significant at alpha=0.05

    # For emergent selectivity analysis
    receives_input: Optional[np.ndarray] = None  # [n_neurons] bool

    @property
    def n_neurons(self) -> int:
        return len(self.auc)

    @property
    def n_significant(self) -> int:
        return int(self.significant.sum())

    @property
    def frac_significant(self) -> float:
        return float(self.significant.mean())

    @property
    def mean_auc(self) -> float:
        return float(np.nanmean(self.auc))

    @property
    def mean_auc_significant(self) -> float:
        if self.n_significant > 0:
            return float(np.nanmean(self.auc[self.significant]))
        return 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'factor_name': self.factor_name,
            'auc': self.auc.tolist(),
            'p_values': self.p_values.tolist(),
            'ci_lower': self.ci_lower.tolist(),
            'ci_upper': self.ci_upper.tolist(),
            'significant': self.significant.tolist(),
            'n_significant': self.n_significant,
            'frac_significant': self.frac_significant,
            'mean_auc': self.mean_auc,
            'receives_input': self.receives_input.tolist() if self.receives_input is not None else None
        }


def compute_roc_auc(rates_cond1: np.ndarray, rates_cond2: np.ndarray) -> float:
    """
    Compute ROC-AUC using Mann-Whitney U statistic.

    AUC = P(X_cond2 > X_cond1) = U / (n1 * n2)

    This matches the implementation in arrayROC.m using rank-based computation.

    Args:
        rates_cond1: [n_trials_cond1] firing rates for condition 1 (e.g., low reward)
        rates_cond2: [n_trials_cond2] firing rates for condition 2 (e.g., high reward)

    Returns:
        AUC in [0, 1], where 0.5 = chance
        AUC > 0.5 means condition 2 tends to have higher rates
        AUC < 0.5 means condition 1 tends to have higher rates
    """
    n1 = len(rates_cond1)
    n2 = len(rates_cond2)

    if n1 == 0 or n2 == 0:
        return np.nan

    # Pool and rank all values
    pooled = np.concatenate([rates_cond1, rates_cond2])
    ranks = stats.rankdata(pooled)

    # Sum of ranks for condition 2
    rank_sum_2 = ranks[n1:].sum()

    # Mann-Whitney U for condition 2
    U2 = rank_sum_2 - n2 * (n2 + 1) / 2

    # AUC = U / (n1 * n2)
    auc = U2 / (n1 * n2)

    return auc


def permutation_test_roc(
    rates_cond1: np.ndarray,
    rates_cond2: np.ndarray,
    observed_auc: float,
    n_perm: int = 200
) -> float:
    """
    Permutation test for ROC-AUC significance.

    Shuffles condition labels and recomputes AUC to build null distribution.

    Args:
        rates_cond1: [n_trials_cond1] firing rates for condition 1
        rates_cond2: [n_trials_cond2] firing rates for condition 2
        observed_auc: The observed AUC value
        n_perm: Number of permutations

    Returns:
        Two-sided p-value
    """
    pooled = np.concatenate([rates_cond1, rates_cond2])
    n1 = len(rates_cond1)
    n_total = len(pooled)

    null_aucs = np.zeros(n_perm)
    for i in range(n_perm):
        perm_idx = np.random.permutation(n_total)
        perm_auc = compute_roc_auc(pooled[perm_idx[:n1]], pooled[perm_idx[n1:]])
        null_aucs[i] = perm_auc

    # Two-sided p-value: how often is null as extreme as observed?
    observed_deviation = abs(observed_auc - 0.5)
    null_deviations = np.abs(null_aucs - 0.5)

    # Add 1 to numerator and denominator for proper p-value estimation
    p_value = (np.sum(null_deviations >= observed_deviation) + 1) / (n_perm + 1)

    return p_value


def bootstrap_ci_roc(
    rates_cond1: np.ndarray,
    rates_cond2: np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for ROC-AUC.

    Resamples each condition with replacement and recomputes AUC.

    Args:
        rates_cond1: [n_trials_cond1] firing rates for condition 1
        rates_cond2: [n_trials_cond2] firing rates for condition 2
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (0.05 for 95% CI)

    Returns:
        (ci_lower, ci_upper) - Bootstrap confidence interval
    """
    n1 = len(rates_cond1)
    n2 = len(rates_cond2)

    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan)

    boot_aucs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample each condition independently with replacement
        idx1 = np.random.randint(0, n1, size=n1)
        idx2 = np.random.randint(0, n2, size=n2)
        boot_auc = compute_roc_auc(rates_cond1[idx1], rates_cond2[idx2])
        boot_aucs[i] = boot_auc

    ci_lower = np.percentile(boot_aucs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_aucs, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


def find_visual_epoch_bins(
    target_onset_bin: int,
    bin_size_ms: float = 25.0,
    window_start_ms: float = 50.0,
    window_end_ms: float = 250.0
) -> Tuple[int, int]:
    """
    Find the bin indices for the visual epoch window.

    Matches the SC visual epoch in neuronal analysis: 50-250ms post-target onset.

    Args:
        target_onset_bin: Bin index of target onset
        bin_size_ms: Size of each bin in ms
        window_start_ms: Start of window relative to target onset (ms)
        window_end_ms: End of window relative to target onset (ms)

    Returns:
        (start_bin, end_bin) - Bin indices for the analysis window
    """
    start_offset = int(window_start_ms / bin_size_ms)
    end_offset = int(window_end_ms / bin_size_ms)

    start_bin = target_onset_bin + start_offset
    end_bin = target_onset_bin + end_offset

    return start_bin, end_bin


def compute_roc_selectivity(
    model_rates: np.ndarray,
    factor_values: np.ndarray,
    epoch_bins: Optional[Tuple[int, int]] = None,
    min_trials: int = 5,
    n_perm: int = 200,
    n_bootstrap: int = 200,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC-AUC selectivity for all neurons.

    For binary factors (0/1), uses those directly as conditions.
    For non-binary factors, splits at median.

    Args:
        model_rates: [n_trials, n_bins, n_neurons] firing rates
        factor_values: [n_trials] factor value per trial
        epoch_bins: (start, end) bin indices for analysis window
        min_trials: Minimum trials per condition
        n_perm: Number of permutations for p-value
        n_bootstrap: Number of bootstrap samples for CI
        alpha: Significance level

    Returns:
        auc: [n_neurons] AUC values
        p_values: [n_neurons] permutation p-values
        ci_lower: [n_neurons] CI lower bounds
        ci_upper: [n_neurons] CI upper bounds
        n_trials_1: [n_neurons] trial count condition 1
        n_trials_2: [n_neurons] trial count condition 2
    """
    n_trials, n_bins, n_neurons = model_rates.shape

    # Average within epoch
    if epoch_bins is not None:
        start, end = epoch_bins
        start = max(0, start)
        end = min(n_bins, end)
        epoch_rates = model_rates[:, start:end, :].mean(axis=1)  # [n_trials, n_neurons]
    else:
        epoch_rates = model_rates.mean(axis=1)

    # Determine conditions
    unique_vals = np.unique(factor_values[~np.isnan(factor_values)])
    if len(unique_vals) == 2:
        # Binary factor
        cond1_mask = factor_values == unique_vals[0]
        cond2_mask = factor_values == unique_vals[1]
    else:
        # Continuous/multi-level: split at median
        median_val = np.median(factor_values)
        cond1_mask = factor_values <= median_val
        cond2_mask = factor_values > median_val

    # Initialize outputs
    auc = np.full(n_neurons, np.nan)
    p_values = np.full(n_neurons, np.nan)
    ci_lower = np.full(n_neurons, np.nan)
    ci_upper = np.full(n_neurons, np.nan)
    n_trials_1 = np.zeros(n_neurons, dtype=int)
    n_trials_2 = np.zeros(n_neurons, dtype=int)

    for i in range(n_neurons):
        rates_1 = epoch_rates[cond1_mask, i]
        rates_2 = epoch_rates[cond2_mask, i]

        # Remove NaNs
        rates_1 = rates_1[~np.isnan(rates_1)]
        rates_2 = rates_2[~np.isnan(rates_2)]

        n_trials_1[i] = len(rates_1)
        n_trials_2[i] = len(rates_2)

        # Check minimum trials
        if len(rates_1) < min_trials or len(rates_2) < min_trials:
            continue

        # Compute AUC
        auc[i] = compute_roc_auc(rates_1, rates_2)

        # Permutation p-value
        p_values[i] = permutation_test_roc(rates_1, rates_2, auc[i], n_perm)

        # Bootstrap CI
        ci_lower[i], ci_upper[i] = bootstrap_ci_roc(rates_1, rates_2, n_bootstrap, alpha)

    return auc, p_values, ci_lower, ci_upper, n_trials_1, n_trials_2


def compute_factor_roc_selectivity(
    model_rates: np.ndarray,
    factor_values: np.ndarray,
    factor_name: str,
    epoch_bins: Optional[Tuple[int, int]] = None,
    receives_input: Optional[np.ndarray] = None,
    min_trials: int = 5,
    n_perm: int = 200,
    n_bootstrap: int = 200,
    alpha: float = 0.05
) -> ROCSelectivityResult:
    """
    Compute ROC selectivity for a single factor.

    Args:
        model_rates: [n_trials, n_bins, n_neurons] firing rates
        factor_values: [n_trials] factor value per trial
        factor_name: Name of the factor
        epoch_bins: (start, end) analysis window
        receives_input: [n_neurons] bool - which neurons receive this factor
        min_trials: Minimum trials per condition
        n_perm: Number of permutations
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        ROCSelectivityResult with AUC, p-values, CIs, and significance
    """
    auc, p_values, ci_lower, ci_upper, n_trials_1, n_trials_2 = compute_roc_selectivity(
        model_rates, factor_values, epoch_bins, min_trials, n_perm, n_bootstrap, alpha
    )

    # Significance: p < alpha AND CI doesn't include 0.5
    significant = (p_values < alpha) & ((ci_lower > 0.5) | (ci_upper < 0.5))

    return ROCSelectivityResult(
        factor_name=factor_name,
        auc=auc,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_trials_cond1=n_trials_1,
        n_trials_cond2=n_trials_2,
        significant=significant,
        receives_input=receives_input
    )


def compute_all_roc_selectivities(
    model_rates: np.ndarray,
    trial_factors: Dict[str, np.ndarray],
    epoch_bins: Optional[Tuple[int, int]] = None,
    assignment: Optional['InputAssignment'] = None,
    min_trials: int = 5,
    n_perm: int = 200,
    n_bootstrap: int = 200
) -> Dict[str, ROCSelectivityResult]:
    """
    Compute ROC selectivity for all factors.

    Args:
        model_rates: [n_trials, n_bins, n_neurons] firing rates
        trial_factors: Dict mapping factor names to [n_trials] arrays
        epoch_bins: (start, end) analysis window
        assignment: InputAssignment for identifying direct receivers
        min_trials: Minimum trials per condition
        n_perm: Number of permutations
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dict mapping factor names to ROCSelectivityResult
    """
    results = {}

    for factor_name, factor_values in trial_factors.items():
        if factor_name not in ['reward', 'probability', 'salience', 'identity']:
            continue

        # Get receiver mask if assignment provided
        receives_input = None
        if assignment is not None:
            receives_input = assignment.get_input_matrix(factor_name)

        results[factor_name] = compute_factor_roc_selectivity(
            model_rates=model_rates,
            factor_values=factor_values,
            factor_name=factor_name,
            epoch_bins=epoch_bins,
            receives_input=receives_input,
            min_trials=min_trials,
            n_perm=n_perm,
            n_bootstrap=n_bootstrap
        )

    return results


def compute_emergent_roc_selectivity(
    roc_result: ROCSelectivityResult
) -> Dict[str, Any]:
    """
    Analyze emergent selectivity from ROC results.

    Emergent selectivity = significant selectivity in neurons that DON'T
    receive direct input for that factor.

    Args:
        roc_result: ROCSelectivityResult with receives_input set

    Returns:
        Dict with emergent selectivity statistics
    """
    if roc_result.receives_input is None:
        return {
            'n_non_receivers': 0,
            'n_emergent_significant': 0,
            'frac_emergent_significant': 0.0,
            'mean_emergent_auc': 0.5,
            'emergent_auc_values': []
        }

    non_receiver_mask = ~roc_result.receives_input
    n_non_receivers = non_receiver_mask.sum()

    if n_non_receivers == 0:
        return {
            'n_non_receivers': 0,
            'n_emergent_significant': 0,
            'frac_emergent_significant': 0.0,
            'mean_emergent_auc': 0.5,
            'emergent_auc_values': []
        }

    emergent_auc = roc_result.auc[non_receiver_mask]
    emergent_significant = roc_result.significant[non_receiver_mask]

    n_emergent_significant = emergent_significant.sum()
    frac_emergent_significant = n_emergent_significant / n_non_receivers
    mean_emergent_auc = np.nanmean(emergent_auc)

    return {
        'n_non_receivers': int(n_non_receivers),
        'n_emergent_significant': int(n_emergent_significant),
        'frac_emergent_significant': float(frac_emergent_significant),
        'mean_emergent_auc': float(mean_emergent_auc),
        'emergent_auc_values': emergent_auc[~np.isnan(emergent_auc)].tolist()
    }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing ROC selectivity module...")

    np.random.seed(42)

    n_trials = 200
    n_bins = 50
    n_neurons = 20

    # Create synthetic rates
    model_rates = np.random.randn(n_trials, n_bins, n_neurons) * 5 + 15

    # Add selectivity to some neurons
    trial_reward = np.random.randint(0, 2, n_trials).astype(float)
    for i in range(10):
        # Neurons 0-9 have reward selectivity
        model_rates[trial_reward == 1, 20:40, i] += 5

    # Compute ROC selectivity
    auc, p_vals, ci_lo, ci_hi, n1, n2 = compute_roc_selectivity(
        model_rates, trial_reward, epoch_bins=(20, 40)
    )

    print(f"\nAUC values (first 10 neurons):")
    for i in range(10):
        print(f"  Neuron {i}: AUC={auc[i]:.3f}, p={p_vals[i]:.3f}, CI=[{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]")

    print(f"\nNeurons with AUC > 0.6: {(auc > 0.6).sum()}")
    print(f"Neurons with p < 0.05: {(p_vals < 0.05).sum()}")

    # Test full factor computation
    trial_factors = {
        'reward': trial_reward,
        'probability': np.random.randint(0, 2, n_trials).astype(float)
    }

    results = compute_all_roc_selectivities(model_rates, trial_factors, epoch_bins=(20, 40))

    print("\n--- Factor Results ---")
    for factor_name, result in results.items():
        print(f"\n{factor_name}:")
        print(f"  Mean AUC: {result.mean_auc:.3f}")
        print(f"  Significant: {result.n_significant}/{result.n_neurons}")

    print("\nAll tests passed!")
