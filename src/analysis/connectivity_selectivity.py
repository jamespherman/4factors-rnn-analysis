"""
Connectivity-Selectivity Analysis Module.

Analyzes the relationship between recurrent connection weights (E->E, I->E, etc.)
and factor selectivity in the trained RNN model.

Based on literature approaches:
- Weight clustering analysis (Song et al. 2016)
- Effective connectivity regression (Mastrogiuseppe & Ostojic 2018)
- Graded ablation analysis
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import torch


@dataclass
class WeightSelectivityCorrelation:
    """Results from weight-selectivity correlation analysis."""
    connection_type: str  # 'EE', 'EI', 'IE', 'II'
    factor_name: str
    correlation: float
    p_value: float
    n_pairs: int
    weight_values: np.ndarray
    selectivity_similarity: np.ndarray


@dataclass
class EffectiveConnectivityResult:
    """Results from effective connectivity regression."""
    factor_name: str
    beta_direct: float      # Coefficient for direct input
    beta_E: float           # Coefficient for E->neuron connections
    beta_I: float           # Coefficient for I->neuron connections
    r_squared: float        # Model R^2
    residuals: np.ndarray   # [n_neurons]
    predicted: np.ndarray   # [n_neurons]
    actual: np.ndarray      # [n_neurons]


@dataclass
class GradedAblationResult:
    """Results from graded ablation analysis."""
    ablation_type: str      # 'EE', 'IE', 'EI', 'II'
    scale_factors: np.ndarray  # [n_levels] scaling factors tested
    selectivity_by_factor: Dict[str, np.ndarray]  # factor -> [n_levels] mean selectivity


def extract_weight_blocks(
    W_rec: np.ndarray,
    n_exc: int
) -> Dict[str, np.ndarray]:
    """
    Extract E->E, E->I, I->E, I->I blocks from recurrent weight matrix.

    Args:
        W_rec: [n_total, n_total] recurrent weight matrix
        n_exc: Number of excitatory neurons

    Returns:
        Dict with 'EE', 'EI', 'IE', 'II' weight blocks
    """
    n_total = W_rec.shape[0]
    n_inh = n_total - n_exc

    return {
        'EE': W_rec[:n_exc, :n_exc],           # E to E
        'EI': W_rec[:n_exc, n_exc:],           # I to E (negative weights)
        'IE': W_rec[n_exc:, :n_exc],           # E to I
        'II': W_rec[n_exc:, n_exc:]            # I to I (negative weights)
    }


def analyze_weight_selectivity_correlation(
    W_rec: np.ndarray,
    selectivity: np.ndarray,
    n_exc: int,
    connection_type: str = 'EE'
) -> WeightSelectivityCorrelation:
    """
    Analyze correlation between connection weights and selectivity similarity.

    For each pair of neurons (i, j), computes:
    - Weight: |W_rec[i, j]| (connection from j to i)
    - Selectivity similarity: -|selectivity[i] - selectivity[j]|

    Positive correlation means neurons with similar selectivity have stronger connections.

    Args:
        W_rec: [n_total, n_total] recurrent weight matrix
        selectivity: [n_total] selectivity values (e.g., AUC for a factor)
        n_exc: Number of excitatory neurons
        connection_type: 'EE', 'EI', 'IE', or 'II'

    Returns:
        WeightSelectivityCorrelation result
    """
    n_total = W_rec.shape[0]

    # Define source and target indices based on connection type
    if connection_type == 'EE':
        targets = np.arange(n_exc)
        sources = np.arange(n_exc)
    elif connection_type == 'EI':  # I -> E
        targets = np.arange(n_exc)
        sources = np.arange(n_exc, n_total)
    elif connection_type == 'IE':  # E -> I
        targets = np.arange(n_exc, n_total)
        sources = np.arange(n_exc)
    elif connection_type == 'II':
        targets = np.arange(n_exc, n_total)
        sources = np.arange(n_exc, n_total)
    else:
        raise ValueError(f"Unknown connection_type: {connection_type}")

    # Collect all pairs
    weights = []
    sel_similarities = []

    for i in targets:
        for j in sources:
            if i == j:
                continue  # Skip self-connections

            w = np.abs(W_rec[i, j])
            if w < 1e-10:
                continue  # Skip zero weights

            # Selectivity similarity (negative distance)
            sel_sim = -np.abs(selectivity[i] - selectivity[j])

            weights.append(w)
            sel_similarities.append(sel_sim)

    weights = np.array(weights)
    sel_similarities = np.array(sel_similarities)

    if len(weights) < 3:
        return WeightSelectivityCorrelation(
            connection_type=connection_type,
            factor_name='',
            correlation=np.nan,
            p_value=1.0,
            n_pairs=len(weights),
            weight_values=weights,
            selectivity_similarity=sel_similarities
        )

    # Pearson correlation
    r, p = stats.pearsonr(weights, sel_similarities)

    return WeightSelectivityCorrelation(
        connection_type=connection_type,
        factor_name='',
        correlation=r if not np.isnan(r) else 0.0,
        p_value=p if not np.isnan(p) else 1.0,
        n_pairs=len(weights),
        weight_values=weights,
        selectivity_similarity=sel_similarities
    )


def analyze_all_weight_selectivity_correlations(
    W_rec: np.ndarray,
    selectivity_by_factor: Dict[str, np.ndarray],
    n_exc: int
) -> Dict[str, Dict[str, WeightSelectivityCorrelation]]:
    """
    Analyze weight-selectivity correlations for all connection types and factors.

    Args:
        W_rec: [n_total, n_total] recurrent weight matrix
        selectivity_by_factor: Dict mapping factor names to [n_total] selectivity arrays
        n_exc: Number of excitatory neurons

    Returns:
        Nested dict: connection_type -> factor_name -> WeightSelectivityCorrelation
    """
    results = {}

    for conn_type in ['EE', 'EI', 'IE', 'II']:
        results[conn_type] = {}

        for factor_name, selectivity in selectivity_by_factor.items():
            result = analyze_weight_selectivity_correlation(
                W_rec, selectivity, n_exc, conn_type
            )
            result.factor_name = factor_name
            results[conn_type][factor_name] = result

    return results


def compute_input_weighted_selectivity(
    W_rec: np.ndarray,
    selectivity: np.ndarray,
    n_exc: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted sum of input selectivity for each neuron.

    For each neuron i:
    - E_input[i] = sum(|W_rec[i, j]| * selectivity[j]) for j in E neurons
    - I_input[i] = sum(|W_rec[i, j]| * selectivity[j]) for j in I neurons

    Args:
        W_rec: [n_total, n_total] recurrent weight matrix
        selectivity: [n_total] selectivity values
        n_exc: Number of excitatory neurons

    Returns:
        E_input: [n_total] weighted E input selectivity
        I_input: [n_total] weighted I input selectivity
    """
    n_total = W_rec.shape[0]

    # Weighted input from E neurons (columns 0:n_exc)
    W_from_E = np.abs(W_rec[:, :n_exc])
    E_input = W_from_E @ selectivity[:n_exc]

    # Weighted input from I neurons (columns n_exc:)
    W_from_I = np.abs(W_rec[:, n_exc:])
    I_input = W_from_I @ selectivity[n_exc:]

    return E_input, I_input


def fit_effective_connectivity_model(
    W_rec: np.ndarray,
    selectivity: np.ndarray,
    receives_input: np.ndarray,
    n_exc: int,
    factor_name: str = ''
) -> EffectiveConnectivityResult:
    """
    Fit effective connectivity model to predict selectivity.

    Model: selectivity[i] = β0 + β_direct * receives_input[i]
                            + β_E * E_input_selectivity[i]
                            + β_I * I_input_selectivity[i]

    Args:
        W_rec: [n_total, n_total] recurrent weight matrix
        selectivity: [n_total] observed selectivity values
        receives_input: [n_total] bool array indicating direct input receivers
        n_exc: Number of excitatory neurons
        factor_name: Name of the factor being analyzed

    Returns:
        EffectiveConnectivityResult with fitted coefficients
    """
    n_total = len(selectivity)

    # Compute weighted input selectivity
    E_input, I_input = compute_input_weighted_selectivity(W_rec, selectivity, n_exc)

    # Build design matrix
    X = np.column_stack([
        np.ones(n_total),           # Intercept
        receives_input.astype(float),  # Direct input
        E_input,                     # E-weighted input
        I_input                      # I-weighted input
    ])

    # Handle NaNs
    valid_mask = ~np.isnan(selectivity)
    X_valid = X[valid_mask]
    y_valid = selectivity[valid_mask]

    if len(y_valid) < 5:
        return EffectiveConnectivityResult(
            factor_name=factor_name,
            beta_direct=np.nan,
            beta_E=np.nan,
            beta_I=np.nan,
            r_squared=0.0,
            residuals=np.full(n_total, np.nan),
            predicted=np.full(n_total, np.nan),
            actual=selectivity.copy()
        )

    # Fit linear regression
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X_valid, y_valid, rcond=None)

        # Predictions
        predicted = np.full(n_total, np.nan)
        predicted[valid_mask] = X_valid @ beta

        # R-squared
        ss_res = np.sum((y_valid - predicted[valid_mask]) ** 2)
        ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        residuals_full = np.full(n_total, np.nan)
        residuals_full[valid_mask] = y_valid - predicted[valid_mask]

        return EffectiveConnectivityResult(
            factor_name=factor_name,
            beta_direct=float(beta[1]),
            beta_E=float(beta[2]),
            beta_I=float(beta[3]),
            r_squared=float(r_squared),
            residuals=residuals_full,
            predicted=predicted,
            actual=selectivity.copy()
        )

    except Exception as e:
        return EffectiveConnectivityResult(
            factor_name=factor_name,
            beta_direct=np.nan,
            beta_E=np.nan,
            beta_I=np.nan,
            r_squared=0.0,
            residuals=np.full(n_total, np.nan),
            predicted=np.full(n_total, np.nan),
            actual=selectivity.copy()
        )


def graded_ablation_selectivity(
    model,
    val_inputs: torch.Tensor,
    trial_factors: Dict[str, np.ndarray],
    epoch_bins: Tuple[int, int],
    ablation_type: str,
    scale_factors: np.ndarray = None,
    device: str = 'cpu'
) -> GradedAblationResult:
    """
    Analyze how selectivity changes with graded ablation of connection types.

    Args:
        model: Trained EIRNN model
        val_inputs: [n_trials, n_bins, n_inputs] validation inputs
        trial_factors: Dict mapping factor names to [n_trials] arrays
        epoch_bins: (start, end) bins for selectivity computation
        ablation_type: 'EE', 'IE', 'EI', or 'II'
        scale_factors: Array of scaling factors to test (default: [0, 0.25, 0.5, 0.75, 1.0])
        device: Device for computation

    Returns:
        GradedAblationResult with selectivity at each ablation level
    """
    from src.analysis.roc_selectivity import compute_roc_selectivity

    if scale_factors is None:
        scale_factors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    n_exc = model.n_exc
    n_total = model.n_total

    # Get original weights
    W_rec_original = model.W_rec.detach().cpu().numpy().copy()

    # Define which block to ablate
    if ablation_type == 'EE':
        row_slice = slice(0, n_exc)
        col_slice = slice(0, n_exc)
    elif ablation_type == 'IE':  # E->I
        row_slice = slice(n_exc, n_total)
        col_slice = slice(0, n_exc)
    elif ablation_type == 'EI':  # I->E
        row_slice = slice(0, n_exc)
        col_slice = slice(n_exc, n_total)
    elif ablation_type == 'II':
        row_slice = slice(n_exc, n_total)
        col_slice = slice(n_exc, n_total)
    else:
        raise ValueError(f"Unknown ablation_type: {ablation_type}")

    # Store results
    selectivity_by_factor = {f: [] for f in trial_factors.keys()
                            if f in ['reward', 'probability', 'salience', 'identity']}

    for scale in scale_factors:
        # Create scaled weight matrix
        W_scaled = W_rec_original.copy()
        W_scaled[row_slice, col_slice] *= scale

        # Temporarily set model weights
        with torch.no_grad():
            # We need to modify the raw weights to achieve the scaled W_rec
            # W_rec = |W_rec_raw| * sign_mask * diag_mask
            # So we set W_rec_raw = scaled_abs_values
            W_raw_scaled = np.abs(W_scaled)
            model.W_rec_raw.data = torch.tensor(W_raw_scaled, dtype=torch.float32, device=device)

        # Run model
        model.eval()
        with torch.no_grad():
            inputs_d = val_inputs.to(device)
            rates, _ = model(inputs_d)
            rates_np = rates.cpu().numpy()

        # Compute selectivity for each factor
        for factor_name, factor_values in trial_factors.items():
            if factor_name not in selectivity_by_factor:
                continue

            auc, _, _, _, _, _ = compute_roc_selectivity(
                rates_np, factor_values, epoch_bins, min_trials=5, n_perm=50
            )

            # Mean AUC deviation from chance
            mean_auc_deviation = np.nanmean(np.abs(auc - 0.5))
            selectivity_by_factor[factor_name].append(mean_auc_deviation)

    # Restore original weights
    with torch.no_grad():
        model.W_rec_raw.data = torch.tensor(np.abs(W_rec_original), dtype=torch.float32, device=device)

    # Convert lists to arrays
    selectivity_by_factor = {k: np.array(v) for k, v in selectivity_by_factor.items()}

    return GradedAblationResult(
        ablation_type=ablation_type,
        scale_factors=scale_factors,
        selectivity_by_factor=selectivity_by_factor
    )


def compute_selectivity_flow(
    W_rec: np.ndarray,
    selectivity: np.ndarray,
    receives_input: np.ndarray,
    n_exc: int,
    threshold_percentile: float = 75.0
) -> Dict[str, Any]:
    """
    Analyze how selectivity "flows" from direct receivers to emergent neurons.

    Identifies strong connections from neurons with direct input (sources)
    to neurons with emergent selectivity (targets).

    Args:
        W_rec: [n_total, n_total] recurrent weight matrix
        selectivity: [n_total] selectivity values
        receives_input: [n_total] bool - direct input receivers
        n_exc: Number of excitatory neurons
        threshold_percentile: Percentile for "strong" connections

    Returns:
        Dict with flow analysis results
    """
    n_total = len(selectivity)

    # Source neurons: receive direct input and have significant selectivity
    sources = receives_input & (np.abs(selectivity - 0.5) > 0.1)

    # Target neurons: don't receive input but have significant selectivity (emergent)
    targets = ~receives_input & (np.abs(selectivity - 0.5) > 0.1)

    # Weight threshold
    nonzero_weights = np.abs(W_rec[W_rec != 0])
    if len(nonzero_weights) == 0:
        weight_threshold = 0
    else:
        weight_threshold = np.percentile(nonzero_weights, threshold_percentile)

    # Find strong connections from sources to targets
    connections = []
    for i in np.where(targets)[0]:
        for j in np.where(sources)[0]:
            w = np.abs(W_rec[i, j])
            if w >= weight_threshold:
                # Determine connection type
                source_is_E = j < n_exc
                target_is_E = i < n_exc
                conn_type = ('E' if source_is_E else 'I') + '->' + ('E' if target_is_E else 'I')

                connections.append({
                    'source': int(j),
                    'target': int(i),
                    'weight': float(w),
                    'source_selectivity': float(selectivity[j]),
                    'target_selectivity': float(selectivity[i]),
                    'connection_type': conn_type
                })

    # Summarize by connection type
    conn_type_counts = {}
    conn_type_weights = {}
    for conn in connections:
        ct = conn['connection_type']
        if ct not in conn_type_counts:
            conn_type_counts[ct] = 0
            conn_type_weights[ct] = []
        conn_type_counts[ct] += 1
        conn_type_weights[ct].append(conn['weight'])

    conn_type_mean_weights = {k: np.mean(v) for k, v in conn_type_weights.items()}

    return {
        'n_sources': int(sources.sum()),
        'n_targets': int(targets.sum()),
        'n_connections': len(connections),
        'connections': connections,
        'connection_type_counts': conn_type_counts,
        'connection_type_mean_weights': conn_type_mean_weights,
        'weight_threshold': float(weight_threshold)
    }


def summarize_connectivity_selectivity_analysis(
    weight_sel_correlations: Dict[str, Dict[str, WeightSelectivityCorrelation]],
    effective_connectivity: Dict[str, EffectiveConnectivityResult],
    selectivity_flow: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create summary of connectivity-selectivity analysis.

    Args:
        weight_sel_correlations: Results from analyze_all_weight_selectivity_correlations
        effective_connectivity: Results from fit_effective_connectivity_model for each factor
        selectivity_flow: Results from compute_selectivity_flow for each factor

    Returns:
        Summary dict for reporting
    """
    summary = {
        'weight_selectivity_correlations': {},
        'effective_connectivity': {},
        'selectivity_flow': {}
    }

    # Weight-selectivity correlations
    for conn_type, factor_results in weight_sel_correlations.items():
        summary['weight_selectivity_correlations'][conn_type] = {}
        for factor_name, result in factor_results.items():
            summary['weight_selectivity_correlations'][conn_type][factor_name] = {
                'correlation': result.correlation,
                'p_value': result.p_value,
                'n_pairs': result.n_pairs,
                'significant': result.p_value < 0.05
            }

    # Effective connectivity
    for factor_name, result in effective_connectivity.items():
        summary['effective_connectivity'][factor_name] = {
            'beta_direct': result.beta_direct,
            'beta_E': result.beta_E,
            'beta_I': result.beta_I,
            'r_squared': result.r_squared
        }

    # Selectivity flow
    for factor_name, flow in selectivity_flow.items():
        summary['selectivity_flow'][factor_name] = {
            'n_sources': flow['n_sources'],
            'n_targets': flow['n_targets'],
            'n_connections': flow['n_connections'],
            'connection_type_counts': flow['connection_type_counts']
        }

    return summary


if __name__ == "__main__":
    print("Testing connectivity-selectivity module...")

    np.random.seed(42)

    # Synthetic network
    n_exc = 30
    n_inh = 10
    n_total = n_exc + n_inh

    # Create random weight matrix with Dale's law
    W_rec = np.random.gamma(2, 0.05, (n_total, n_total))
    W_rec[:, n_exc:] *= -1  # I columns negative
    np.fill_diagonal(W_rec, 0)

    # Synthetic selectivity (clustered)
    selectivity = np.zeros(n_total)
    selectivity[:15] = 0.7 + 0.1 * np.random.randn(15)  # High selectivity cluster
    selectivity[15:30] = 0.3 + 0.1 * np.random.randn(15)  # Low selectivity cluster
    selectivity[30:] = 0.5 + 0.1 * np.random.randn(10)  # I neurons near chance

    selectivity = np.clip(selectivity, 0, 1)

    # Test weight-selectivity correlation
    result = analyze_weight_selectivity_correlation(W_rec, selectivity, n_exc, 'EE')
    print(f"\nE->E correlation: r={result.correlation:.3f}, p={result.p_value:.3f}, n={result.n_pairs}")

    # Test all correlations
    selectivity_by_factor = {'reward': selectivity}
    all_results = analyze_all_weight_selectivity_correlations(W_rec, selectivity_by_factor, n_exc)

    print("\n--- All correlations ---")
    for conn_type, factor_results in all_results.items():
        for factor_name, res in factor_results.items():
            print(f"  {conn_type}/{factor_name}: r={res.correlation:.3f}, p={res.p_value:.3f}")

    # Test effective connectivity
    receives_input = np.zeros(n_total, dtype=bool)
    receives_input[:20] = True

    ec_result = fit_effective_connectivity_model(W_rec, selectivity, receives_input, n_exc, 'reward')
    print(f"\n--- Effective connectivity ---")
    print(f"  β_direct: {ec_result.beta_direct:.4f}")
    print(f"  β_E: {ec_result.beta_E:.4f}")
    print(f"  β_I: {ec_result.beta_I:.4f}")
    print(f"  R²: {ec_result.r_squared:.4f}")

    # Test selectivity flow
    flow = compute_selectivity_flow(W_rec, selectivity, receives_input, n_exc)
    print(f"\n--- Selectivity flow ---")
    print(f"  Sources: {flow['n_sources']}")
    print(f"  Targets: {flow['n_targets']}")
    print(f"  Strong connections: {flow['n_connections']}")
    print(f"  By type: {flow['connection_type_counts']}")

    print("\nAll tests passed!")
