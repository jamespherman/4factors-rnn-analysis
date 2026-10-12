"""
Analysis modules for input restructuring experiments.
"""

from .emergent_selectivity import (
    compute_factor_selectivity,
    compute_emergent_selectivity,
    compute_all_factor_selectivities,
    EmergentSelectivityResult,
)

from .ablation import (
    create_ablated_model,
    run_ablation_analysis,
    AblationResult,
)

__all__ = [
    'compute_factor_selectivity',
    'compute_emergent_selectivity',
    'compute_all_factor_selectivities',
    'EmergentSelectivityResult',
    'create_ablated_model',
    'run_ablation_analysis',
    'AblationResult',
]
