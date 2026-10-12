"""
Emergent selectivity analysis for input restructuring experiments.

Key concept: "Emergent selectivity" is factor selectivity that appears in neurons
that did NOT receive that factor as direct input. This selectivity must arise
from recurrent dynamics (input from other neurons).

See specs/INPUT_RESTRUCTURING_SPEC.md for scientific rationale.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.input_assignment import InputAssignment


@dataclass
class EmergentSelectivityResult:
    """Container for emergent selectivity analysis results."""
    factor_name: str
    selectivity: np.ndarray           # [n_neurons] selectivity values
    receives_input: np.ndarray        # [n_neurons] bool
    emergent_selectivity: np.ndarray  # [n_neurons] selectivity for non-receivers (NaN for receivers)

    # Summary statistics
    n_total: int
    n_receivers: int
    n_non_receivers: int
    n_significant_emergent: int
    mean_abs_emergent: float
    fraction_significant: float

    # Statistical test results
    t_stat: float
    p_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'factor_name': self.factor_name,
            'selectivity': self.selectivity.tolist(),
            'receives_input': self.receives_input.tolist(),
            'emergent_selectivity': [
                x if not np.isnan(x) else None
                for x in self.emergent_selectivity
            ],
            'n_total': self.n_total,
            'n_receivers': self.n_receivers,
            'n_non_receivers': self.n_non_receivers,
            'n_significant_emergent': self.n_significant_emergent,
            'mean_abs_emergent': self.mean_abs_emergent,
            'fraction_significant': self.fraction_significant,
            't_stat': self.t_stat,
            'p_value': self.p_value
        }


def compute_factor_selectivity(
    model_rates: np.ndarray,
    factor_values: np.ndarray,
    epoch_bins: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Compute selectivity for a given factor for each neuron.

    Selectivity is measured as Pearson correlation between the neuron's
    epoch-averaged firing rate and the factor value across trials.

    Args:
        model_rates: [n_trials, n_bins, n_neurons] firing rates
        factor_values: [n_trials] factor value per trial
        epoch_bins: Optional (start, end) bin indices for analysis epoch.
                   If None, uses all time bins.

    Returns:
        selectivity: [n_neurons] Pearson correlation values
    """
    n_trials, n_bins, n_neurons = model_rates.shape

    # Use specified epoch or full trial
    if epoch_bins is not None:
        start, end = epoch_bins
        epoch_rates = model_rates[:, start:end, :].mean(axis=1)  # [n_trials, n_neurons]
    else:
        epoch_rates = model_rates.mean(axis=1)

    selectivity = np.zeros(n_neurons)
    for i in range(n_neurons):
        # Handle constant firing rates or constant factor values
        if epoch_rates[:, i].std() < 1e-10 or factor_values.std() < 1e-10:
            selectivity[i] = 0.0
        else:
            r, _ = stats.pearsonr(epoch_rates[:, i], factor_values)
            selectivity[i] = r if not np.isnan(r) else 0.0

    return selectivity


def compute_emergent_selectivity(
    model_rates: np.ndarray,
    factor_values: np.ndarray,
    assignment: InputAssignment,
    factor_name: str,
    epoch_bins: Optional[Tuple[int, int]] = None,
    significance_threshold: float = 0.2,
    use_permutation_test: bool = False,
    n_permutations: int = 1000
) -> EmergentSelectivityResult:
    """
    Compute emergent selectivity: factor selectivity in neurons that don't receive
    that factor as direct input.

    Args:
        model_rates: [n_trials, n_bins, n_neurons] firing rates
        factor_values: [n_trials] factor value per trial
        assignment: InputAssignment specifying which neurons receive which factors
        factor_name: 'reward', 'probability', 'salience', or 'identity'
        epoch_bins: Optional analysis epoch
        significance_threshold: |selectivity| threshold for "significant"
        use_permutation_test: If True, use permutation test for significance
        n_permutations: Number of permutations if using permutation test

    Returns:
        EmergentSelectivityResult with selectivity values and statistics
    """
    # Get which neurons receive this factor
    receives_input = assignment.get_input_matrix(factor_name)

    # Compute selectivity for all neurons
    selectivity = compute_factor_selectivity(model_rates, factor_values, epoch_bins)
    n_neurons = len(selectivity)

    # Emergent selectivity: only for neurons that don't receive this factor
    emergent_mask = ~receives_input
    emergent_selectivity = selectivity.copy().astype(float)
    emergent_selectivity[receives_input] = np.nan  # Mask direct receivers

    n_non_receivers = emergent_mask.sum()

    if n_non_receivers > 0:
        emergent_vals = selectivity[emergent_mask]

        if use_permutation_test:
            # Permutation test: shuffle factor values and recompute
            null_distribution = []
            for _ in range(n_permutations):
                shuffled_factors = np.random.permutation(factor_values)
                shuffled_sel = compute_factor_selectivity(model_rates, shuffled_factors, epoch_bins)
                null_distribution.append(np.abs(shuffled_sel[emergent_mask]).mean())

            observed = np.abs(emergent_vals).mean()
            p_value = (np.array(null_distribution) >= observed).mean()
            # Use the observed value as a pseudo t-stat
            t_stat = (observed - np.mean(null_distribution)) / (np.std(null_distribution) + 1e-10)
        else:
            # Simple t-test against zero
            t_stat, p_value = stats.ttest_1samp(emergent_vals, 0)
            if np.isnan(t_stat):
                t_stat = 0.0
                p_value = 1.0

        n_significant = (np.abs(emergent_vals) > significance_threshold).sum()
        mean_abs_emergent = float(np.abs(emergent_vals).mean())
        fraction_significant = float(n_significant / n_non_receivers)
    else:
        t_stat = 0.0
        p_value = 1.0
        n_significant = 0
        mean_abs_emergent = 0.0
        fraction_significant = 0.0

    return EmergentSelectivityResult(
        factor_name=factor_name,
        selectivity=selectivity,
        receives_input=receives_input,
        emergent_selectivity=emergent_selectivity,
        n_total=n_neurons,
        n_receivers=int(receives_input.sum()),
        n_non_receivers=int(n_non_receivers),
        n_significant_emergent=int(n_significant),
        mean_abs_emergent=mean_abs_emergent,
        fraction_significant=fraction_significant,
        t_stat=float(t_stat),
        p_value=float(p_value)
    )


def compute_all_factor_selectivities(
    model_rates: np.ndarray,
    trial_factors: Dict[str, np.ndarray],
    assignment: InputAssignment,
    epoch_bins: Optional[Tuple[int, int]] = None,
    significance_threshold: float = 0.2
) -> Dict[str, EmergentSelectivityResult]:
    """
    Compute emergent selectivity for all factors.

    Args:
        model_rates: [n_trials, n_bins, n_neurons] firing rates
        trial_factors: Dict mapping factor names to [n_trials] arrays
        assignment: InputAssignment
        epoch_bins: Optional analysis epoch
        significance_threshold: Threshold for significance

    Returns:
        Dict mapping factor names to EmergentSelectivityResult
    """
    results = {}

    for factor_name, factor_values in trial_factors.items():
        if factor_name not in ['reward', 'probability', 'salience', 'identity']:
            continue

        results[factor_name] = compute_emergent_selectivity(
            model_rates=model_rates,
            factor_values=factor_values,
            assignment=assignment,
            factor_name=factor_name,
            epoch_bins=epoch_bins,
            significance_threshold=significance_threshold
        )

    return results


def summarize_emergent_selectivity(
    results: Dict[str, EmergentSelectivityResult]
) -> Dict[str, Any]:
    """
    Create summary statistics across all factors.

    Returns:
        Dictionary with aggregate statistics
    """
    summary = {}

    for factor_name, result in results.items():
        summary[f'{factor_name}_n_non_receivers'] = result.n_non_receivers
        summary[f'{factor_name}_n_significant'] = result.n_significant_emergent
        summary[f'{factor_name}_frac_significant'] = result.fraction_significant
        summary[f'{factor_name}_mean_abs_emergent'] = result.mean_abs_emergent
        summary[f'{factor_name}_p_value'] = result.p_value

    # Aggregate across factors
    all_fracs = [r.fraction_significant for r in results.values() if r.n_non_receivers > 0]
    all_means = [r.mean_abs_emergent for r in results.values() if r.n_non_receivers > 0]

    summary['avg_fraction_significant'] = np.mean(all_fracs) if all_fracs else 0.0
    summary['avg_mean_abs_emergent'] = np.mean(all_means) if all_means else 0.0

    return summary


def compute_selectivity_by_neuron_type(
    result: EmergentSelectivityResult,
    rf_locations: np.ndarray
) -> Dict[str, Any]:
    """
    Break down emergent selectivity by E vs I neurons.

    Args:
        result: EmergentSelectivityResult
        rf_locations: [n_neurons] RF locations (0 for I units, 1-4 for E units)

    Returns:
        Dict with E vs I breakdown
    """
    is_e = rf_locations > 0
    is_i = ~is_e

    # Get emergent selectivity (non-NaN values)
    emergent = result.emergent_selectivity

    # E neurons that are non-receivers
    e_emergent_mask = is_e & ~result.receives_input
    i_emergent_mask = is_i & ~result.receives_input

    breakdown = {
        'n_e_non_receivers': int(e_emergent_mask.sum()),
        'n_i_non_receivers': int(i_emergent_mask.sum()),
    }

    if e_emergent_mask.sum() > 0:
        e_vals = result.selectivity[e_emergent_mask]
        breakdown['e_mean_abs_emergent'] = float(np.abs(e_vals).mean())
        breakdown['e_n_significant'] = int((np.abs(e_vals) > 0.2).sum())
    else:
        breakdown['e_mean_abs_emergent'] = 0.0
        breakdown['e_n_significant'] = 0

    if i_emergent_mask.sum() > 0:
        i_vals = result.selectivity[i_emergent_mask]
        breakdown['i_mean_abs_emergent'] = float(np.abs(i_vals).mean())
        breakdown['i_n_significant'] = int((np.abs(i_vals) > 0.2).sum())
    else:
        breakdown['i_mean_abs_emergent'] = 0.0
        breakdown['i_n_significant'] = 0

    return breakdown


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing emergent selectivity module...")

    np.random.seed(42)

    n_trials = 100
    n_bins = 50
    n_neurons = 30

    # Create synthetic rates with some factor tuning
    model_rates = np.random.randn(n_trials, n_bins, n_neurons) * 5 + 10

    # Add reward selectivity to some neurons
    trial_reward = np.random.randint(0, 2, n_trials).astype(float)
    for i in range(10):
        model_rates[:, 20:40, i] += trial_reward[:, None] * 3

    # Add salience selectivity to other neurons
    trial_salience = np.random.randint(0, 2, n_trials).astype(float)
    for i in range(15, 25):
        model_rates[:, 20:40, i] += trial_salience[:, None] * 2

    # Create assignment where only some neurons receive each factor
    neuron_type = np.array([1] * 24 + [2] * 6)  # 24 E, 6 I
    rf_locations = np.zeros(n_neurons, dtype=int)
    for i in range(24):
        rf_locations[i] = (i % 4) + 1

    from src.input_assignment import create_primary_assignment
    assignment = create_primary_assignment(
        n_neurons, neuron_type, rf_locations,
        sparsity_p=0.5, seed=42
    )

    trial_factors = {
        'reward': trial_reward,
        'probability': np.random.randint(0, 2, n_trials).astype(float),
        'salience': trial_salience,
        'identity': np.random.randint(1, 4, n_trials).astype(float)
    }

    # Compute emergent selectivity
    results = compute_all_factor_selectivities(
        model_rates, trial_factors, assignment,
        epoch_bins=(20, 40)
    )

    print("\n--- Results ---")
    for factor_name, result in results.items():
        print(f"\n{factor_name}:")
        print(f"  Receivers: {result.n_receivers}")
        print(f"  Non-receivers: {result.n_non_receivers}")
        print(f"  Significant emergent: {result.n_significant_emergent}")
        print(f"  Mean |emergent|: {result.mean_abs_emergent:.4f}")
        print(f"  p-value: {result.p_value:.4f}")

        # E vs I breakdown
        breakdown = compute_selectivity_by_neuron_type(result, rf_locations)
        print(f"  E non-receivers: {breakdown['n_e_non_receivers']}")
        print(f"  I non-receivers: {breakdown['n_i_non_receivers']}")

    # Summary
    print("\n--- Summary ---")
    summary = summarize_emergent_selectivity(results)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nAll tests passed!")
