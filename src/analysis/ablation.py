"""
Recurrence ablation analysis for input restructuring experiments.

Tests whether emergent selectivity depends on recurrent connections by:
1. Training a model normally
2. Zeroing recurrent weights (or specific submatrices)
3. Comparing selectivity before vs after ablation

If emergent selectivity drops after ablation, it confirms the selectivity
arose from recurrent dynamics rather than residual direct input.

See specs/INPUT_RESTRUCTURING_SPEC.md for scientific rationale.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.input_assignment import InputAssignment
from src.analysis.emergent_selectivity import (
    compute_emergent_selectivity,
    EmergentSelectivityResult
)


@dataclass
class AblationResult:
    """Container for ablation analysis results."""
    ablation_type: str
    factor_name: str

    # Pre-ablation results
    original_mean_abs_emergent: float
    original_n_significant: int
    original_p_value: float

    # Post-ablation results
    ablated_mean_abs_emergent: float
    ablated_n_significant: int
    ablated_p_value: float

    # Change metrics
    reduction_fraction: float  # How much emergent selectivity decreased
    delta_n_significant: int   # Change in number of significant neurons

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ablation_type': self.ablation_type,
            'factor_name': self.factor_name,
            'original_mean_abs_emergent': self.original_mean_abs_emergent,
            'original_n_significant': self.original_n_significant,
            'original_p_value': self.original_p_value,
            'ablated_mean_abs_emergent': self.ablated_mean_abs_emergent,
            'ablated_n_significant': self.ablated_n_significant,
            'ablated_p_value': self.ablated_p_value,
            'reduction_fraction': self.reduction_fraction,
            'delta_n_significant': self.delta_n_significant
        }


def create_ablated_model(
    model: nn.Module,
    ablation_type: str = 'full_recurrence'
) -> nn.Module:
    """
    Create a copy of the model with specified recurrent weights ablated.

    Args:
        model: Trained EIRNN model
        ablation_type:
            - 'full_recurrence': Zero all W_rec (no recurrent connections)
            - 'ie_only': Zero only I->E connections
            - 'ei_only': Zero only E->I connections
            - 'ee_only': Zero only E->E connections
            - 'ii_only': Zero only I->I connections

    Returns:
        Deep copy of model with specified weights zeroed
    """
    ablated_model = copy.deepcopy(model)
    ablated_model.eval()

    n_exc = model.n_exc
    n_total = model.n_total

    with torch.no_grad():
        if ablation_type == 'full_recurrence':
            # Zero all recurrent weights
            ablated_model.W_rec_raw.zero_()

        elif ablation_type == 'ie_only':
            # I->E: rows 0:n_exc (E targets), cols n_exc:n_total (I sources)
            ablated_model.W_rec_raw[:n_exc, n_exc:] = 0

        elif ablation_type == 'ei_only':
            # E->I: rows n_exc:n_total (I targets), cols 0:n_exc (E sources)
            ablated_model.W_rec_raw[n_exc:, :n_exc] = 0

        elif ablation_type == 'ee_only':
            # E->E: rows 0:n_exc, cols 0:n_exc
            ablated_model.W_rec_raw[:n_exc, :n_exc] = 0

        elif ablation_type == 'ii_only':
            # I->I: rows n_exc:n_total, cols n_exc:n_total
            ablated_model.W_rec_raw[n_exc:, n_exc:] = 0

        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

    return ablated_model


def run_model_and_get_rates(
    model: nn.Module,
    inputs: torch.Tensor,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Run model forward pass and return firing rates as numpy array.

    Args:
        model: EIRNN model
        inputs: [n_trials, n_bins, n_inputs] input tensor
        device: 'cpu' or 'cuda'

    Returns:
        rates: [n_trials, n_bins, n_neurons] numpy array
    """
    model.eval()
    model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        rates, _ = model(inputs)
        return rates.cpu().numpy()


def run_single_ablation(
    model: nn.Module,
    inputs: torch.Tensor,
    factor_values: np.ndarray,
    assignment: InputAssignment,
    factor_name: str,
    ablation_type: str,
    epoch_bins: Optional[Tuple[int, int]] = None,
    device: str = 'cpu',
    n_recorded: Optional[int] = None
) -> AblationResult:
    """
    Run ablation analysis for a single factor and ablation type.

    Args:
        model: Trained EIRNN model
        inputs: [n_trials, n_bins, n_inputs] input tensor
        factor_values: [n_trials] factor values
        assignment: InputAssignment
        factor_name: 'reward', 'probability', 'salience', or 'identity'
        ablation_type: Type of ablation
        epoch_bins: Analysis epoch
        device: Compute device
        n_recorded: Number of recorded neurons (if None, use all)

    Returns:
        AblationResult with before/after comparison
    """
    # Get original model rates
    original_rates = run_model_and_get_rates(model, inputs, device)

    # Limit to recorded neurons if specified
    if n_recorded is not None:
        original_rates = original_rates[:, :, :n_recorded]

    # Compute original emergent selectivity
    original_result = compute_emergent_selectivity(
        original_rates, factor_values, assignment, factor_name, epoch_bins
    )

    # Create ablated model and get rates
    ablated_model = create_ablated_model(model, ablation_type)
    ablated_rates = run_model_and_get_rates(ablated_model, inputs, device)

    # Limit to recorded neurons if specified
    if n_recorded is not None:
        ablated_rates = ablated_rates[:, :, :n_recorded]

    # Compute ablated emergent selectivity
    ablated_result = compute_emergent_selectivity(
        ablated_rates, factor_values, assignment, factor_name, epoch_bins
    )

    # Compute change metrics
    if original_result.mean_abs_emergent > 0:
        reduction = 1 - (ablated_result.mean_abs_emergent / original_result.mean_abs_emergent)
    else:
        reduction = 0.0

    return AblationResult(
        ablation_type=ablation_type,
        factor_name=factor_name,
        original_mean_abs_emergent=original_result.mean_abs_emergent,
        original_n_significant=original_result.n_significant_emergent,
        original_p_value=original_result.p_value,
        ablated_mean_abs_emergent=ablated_result.mean_abs_emergent,
        ablated_n_significant=ablated_result.n_significant_emergent,
        ablated_p_value=ablated_result.p_value,
        reduction_fraction=float(reduction),
        delta_n_significant=ablated_result.n_significant_emergent - original_result.n_significant_emergent
    )


def run_ablation_analysis(
    model: nn.Module,
    inputs: torch.Tensor,
    trial_factors: Dict[str, np.ndarray],
    assignment: InputAssignment,
    epoch_bins: Optional[Tuple[int, int]] = None,
    ablation_types: List[str] = None,
    device: str = 'cpu',
    n_recorded: Optional[int] = None
) -> Dict[str, Dict[str, AblationResult]]:
    """
    Run comprehensive ablation analysis across factors and ablation types.

    Args:
        model: Trained EIRNN model
        inputs: [n_trials, n_bins, n_inputs] input tensor
        trial_factors: Dict mapping factor names to [n_trials] arrays
        assignment: InputAssignment
        epoch_bins: Analysis epoch
        ablation_types: List of ablation types to test
        device: Compute device

    Returns:
        Nested dict: results[factor_name][ablation_type] = AblationResult
    """
    if ablation_types is None:
        ablation_types = ['full_recurrence', 'ie_only']

    results = {}

    for factor_name, factor_values in trial_factors.items():
        if factor_name not in ['reward', 'probability', 'salience', 'identity']:
            continue

        results[factor_name] = {}

        for ablation_type in ablation_types:
            result = run_single_ablation(
                model=model,
                inputs=inputs,
                factor_values=factor_values,
                assignment=assignment,
                factor_name=factor_name,
                ablation_type=ablation_type,
                epoch_bins=epoch_bins,
                device=device,
                n_recorded=n_recorded
            )
            results[factor_name][ablation_type] = result

    return results


def summarize_ablation_results(
    results: Dict[str, Dict[str, AblationResult]]
) -> Dict[str, Any]:
    """
    Create summary statistics across ablation results.

    Returns:
        Dictionary with aggregate statistics
    """
    summary = {
        'by_factor': {},
        'by_ablation': {}
    }

    # Aggregate by factor
    for factor_name, factor_results in results.items():
        factor_summary = {}
        for ablation_type, result in factor_results.items():
            factor_summary[f'{ablation_type}_reduction'] = result.reduction_fraction
        summary['by_factor'][factor_name] = factor_summary

    # Aggregate by ablation type
    all_ablation_types = set()
    for factor_results in results.values():
        all_ablation_types.update(factor_results.keys())

    for ablation_type in all_ablation_types:
        reductions = []
        for factor_results in results.values():
            if ablation_type in factor_results:
                reductions.append(factor_results[ablation_type].reduction_fraction)
        summary['by_ablation'][ablation_type] = {
            'mean_reduction': float(np.mean(reductions)) if reductions else 0.0,
            'min_reduction': float(np.min(reductions)) if reductions else 0.0,
            'max_reduction': float(np.max(reductions)) if reductions else 0.0
        }

    return summary


def print_ablation_report(results: Dict[str, Dict[str, AblationResult]]) -> str:
    """
    Generate human-readable ablation report.

    Returns:
        Formatted string report
    """
    lines = ["=" * 60]
    lines.append("ABLATION ANALYSIS REPORT")
    lines.append("=" * 60)

    for factor_name, factor_results in results.items():
        lines.append(f"\n{factor_name.upper()}")
        lines.append("-" * 40)

        for ablation_type, result in factor_results.items():
            lines.append(f"\n  {ablation_type}:")
            lines.append(f"    Original |emergent|: {result.original_mean_abs_emergent:.4f}")
            lines.append(f"    Ablated |emergent|:  {result.ablated_mean_abs_emergent:.4f}")
            lines.append(f"    Reduction: {result.reduction_fraction * 100:.1f}%")
            lines.append(f"    Significant: {result.original_n_significant} -> {result.ablated_n_significant}")

    lines.append("\n" + "=" * 60)

    # Summary
    summary = summarize_ablation_results(results)
    lines.append("\nSUMMARY BY ABLATION TYPE:")
    for abl_type, stats in summary['by_ablation'].items():
        lines.append(f"  {abl_type}: mean reduction = {stats['mean_reduction']*100:.1f}%")

    return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test with synthetic model
    print("Testing ablation module...")

    import torch.nn as nn

    # Create minimal mock model
    class MockEIRNN(nn.Module):
        def __init__(self, n_exc=20, n_inh=10, n_inputs=14):
            super().__init__()
            self.n_exc = n_exc
            self.n_inh = n_inh
            self.n_total = n_exc + n_inh
            self.W_rec_raw = nn.Parameter(torch.randn(self.n_total, self.n_total) * 0.1)

            # Sign mask for Dale's law
            sign_mask = torch.ones(self.n_total, self.n_total)
            sign_mask[:, n_exc:] = -1
            self.register_buffer('sign_mask', sign_mask)

            self.W_in = nn.Parameter(torch.randn(self.n_total, n_inputs) * 0.1)

        @property
        def W_rec(self):
            return torch.abs(self.W_rec_raw) * self.sign_mask

        def forward(self, inputs):
            batch, time, _ = inputs.shape
            rates = torch.randn(batch, time, self.n_total) * 5 + 10
            outputs = torch.zeros(batch, time, 2)
            return rates, outputs

    # Test ablation
    model = MockEIRNN()

    ablated_full = create_ablated_model(model, 'full_recurrence')
    assert (ablated_full.W_rec_raw == 0).all(), "Full ablation failed"

    ablated_ie = create_ablated_model(model, 'ie_only')
    assert (ablated_ie.W_rec_raw[:model.n_exc, model.n_exc:] == 0).all(), "I->E ablation failed"

    print("  Ablation creation: OK")

    # Test with synthetic data
    np.random.seed(42)
    n_trials, n_bins = 50, 30
    inputs = torch.randn(n_trials, n_bins, 14)

    neuron_type = np.array([1] * 20 + [2] * 10)
    rf_locations = np.zeros(30, dtype=int)
    for i in range(20):
        rf_locations[i] = (i % 4) + 1

    from src.input_assignment import create_primary_assignment
    assignment = create_primary_assignment(30, neuron_type, rf_locations, 0.5, 0)

    trial_factors = {
        'reward': np.random.randint(0, 2, n_trials).astype(float),
        'salience': np.random.randint(0, 2, n_trials).astype(float),
    }

    # Run ablation analysis
    results = run_ablation_analysis(
        model, inputs, trial_factors, assignment,
        ablation_types=['full_recurrence', 'ie_only']
    )

    print("\n" + print_ablation_report(results))

    print("\nAll tests passed!")
