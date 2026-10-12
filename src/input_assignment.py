"""
Input assignment module for restructured input experiments.

Defines which neurons receive which input factors, supporting:
- RF-matched spatial factors (reward, probability) for E units
- Sparse random assignment of feature factors (salience, identity)
- Temporal control (factors from trial start vs target onset)

See specs/INPUT_RESTRUCTURING_SPEC.md for design rationale.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class InputAssignment:
    """
    Tracks which neurons receive which input factors.

    This is the core data structure for input restructuring experiments.
    Each boolean array has shape [n_neurons] and indicates whether that
    neuron receives the corresponding factor as direct input.

    Attributes:
        receives_reward: E units get RF-matched reward, I units get nothing
        receives_probability: E units get RF-matched probability, I units get nothing
        spatial_from_trial_start: If True, spatial factors available from t=0;
                                  if False, from target onset only
        receives_salience: Sparse random assignment of salience input
        receives_identity: Sparse random assignment of identity input
        rf_locations: RF location for each neuron (1-4 for E units, 0 for I units)
        experiment_type: Identifier (E0, E1, E2, E3a, E3b, E4)
        sparsity_p: Probability of receiving each feature factor
        seed: Random seed for sparse assignment (for reproducibility)
    """
    receives_reward: np.ndarray
    receives_probability: np.ndarray
    spatial_from_trial_start: bool
    receives_salience: np.ndarray
    receives_identity: np.ndarray
    rf_locations: np.ndarray
    experiment_type: str
    sparsity_p: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        return {
            'receives_reward': self.receives_reward.tolist(),
            'receives_probability': self.receives_probability.tolist(),
            'spatial_from_trial_start': self.spatial_from_trial_start,
            'receives_salience': self.receives_salience.tolist(),
            'receives_identity': self.receives_identity.tolist(),
            'rf_locations': self.rf_locations.tolist(),
            'experiment_type': self.experiment_type,
            'sparsity_p': self.sparsity_p,
            'seed': self.seed
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InputAssignment':
        """Load from saved dictionary."""
        return cls(
            receives_reward=np.array(d['receives_reward'], dtype=bool),
            receives_probability=np.array(d['receives_probability'], dtype=bool),
            spatial_from_trial_start=d['spatial_from_trial_start'],
            receives_salience=np.array(d['receives_salience'], dtype=bool),
            receives_identity=np.array(d['receives_identity'], dtype=bool),
            rf_locations=np.array(d['rf_locations'], dtype=int),
            experiment_type=d['experiment_type'],
            sparsity_p=d['sparsity_p'],
            seed=d['seed']
        )

    def get_input_matrix(self, factor: str) -> np.ndarray:
        """Get the receives_* array for a given factor name."""
        if factor == 'reward':
            return self.receives_reward
        elif factor == 'probability':
            return self.receives_probability
        elif factor == 'salience':
            return self.receives_salience
        elif factor == 'identity':
            return self.receives_identity
        else:
            raise ValueError(f"Unknown factor: {factor}")

    def summary(self) -> str:
        """Return a human-readable summary."""
        n_neurons = len(self.receives_reward)
        n_e = (self.rf_locations > 0).sum()
        n_i = n_neurons - n_e

        lines = [
            f"InputAssignment: {self.experiment_type}",
            f"  Neurons: {n_neurons} ({n_e} E, {n_i} I)",
            f"  Sparsity p: {self.sparsity_p}",
            f"  Seed: {self.seed}",
            f"  Spatial from trial start: {self.spatial_from_trial_start}",
            f"  Receives reward: {self.receives_reward.sum()} neurons",
            f"  Receives probability: {self.receives_probability.sum()} neurons",
            f"  Receives salience: {self.receives_salience.sum()} neurons",
            f"  Receives identity: {self.receives_identity.sum()} neurons",
        ]
        return '\n'.join(lines)


# =============================================================================
# Factory Functions for Each Experiment Type
# =============================================================================

def create_baseline_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: Optional[np.ndarray] = None
) -> InputAssignment:
    """
    E0: Baseline - all neurons receive all factors at target onset.

    This matches the current (pre-restructuring) approach where every neuron
    gets complete factor information tied to target onset.

    Args:
        n_neurons: Total number of neurons
        neuron_type: [n_neurons] array, 1=E, 2=I
        rf_locations: Optional RF locations (if None, assigns evenly)

    Returns:
        InputAssignment for E0 baseline
    """
    if rf_locations is None:
        # Assign RFs evenly across locations for E units
        rf_locations = np.zeros(n_neurons, dtype=int)
        e_indices = np.where(neuron_type == 1)[0]
        for i, idx in enumerate(e_indices):
            rf_locations[idx] = (i % 4) + 1  # 1-4

    return InputAssignment(
        receives_reward=np.ones(n_neurons, dtype=bool),
        receives_probability=np.ones(n_neurons, dtype=bool),
        spatial_from_trial_start=False,  # At target onset (baseline)
        receives_salience=np.ones(n_neurons, dtype=bool),
        receives_identity=np.ones(n_neurons, dtype=bool),
        rf_locations=rf_locations,
        experiment_type='E0',
        sparsity_p=1.0,
        seed=0
    )


def create_temporal_fix_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: Optional[np.ndarray] = None
) -> InputAssignment:
    """
    E1: Temporal fix - all neurons receive all factors, but spatial
    factors (reward, probability) are available from trial start.

    This tests whether temporal availability alone affects learning.

    Args:
        n_neurons: Total number of neurons
        neuron_type: [n_neurons] array, 1=E, 2=I
        rf_locations: Optional RF locations

    Returns:
        InputAssignment for E1
    """
    if rf_locations is None:
        rf_locations = np.zeros(n_neurons, dtype=int)
        e_indices = np.where(neuron_type == 1)[0]
        for i, idx in enumerate(e_indices):
            rf_locations[idx] = (i % 4) + 1

    return InputAssignment(
        receives_reward=np.ones(n_neurons, dtype=bool),
        receives_probability=np.ones(n_neurons, dtype=bool),
        spatial_from_trial_start=True,  # From trial start
        receives_salience=np.ones(n_neurons, dtype=bool),
        receives_identity=np.ones(n_neurons, dtype=bool),
        rf_locations=rf_locations,
        experiment_type='E1',
        sparsity_p=1.0,
        seed=0
    )


def create_primary_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: np.ndarray,
    sparsity_p: float = 0.5,
    seed: int = 0,
    experiment_type: str = 'E2'
) -> InputAssignment:
    """
    E2/E3/E4: Primary test - RF-matched spatial for E units only,
    sparse random assignment of feature factors.

    This is the key experimental condition that creates a "fair test"
    of recurrence-dependent computation:
    - E units receive spatial factors only for their RF location
    - I units receive NO spatial factor information
    - Feature factors (salience, identity) are randomly assigned

    Args:
        n_neurons: Total number of neurons
        neuron_type: [n_neurons] array, 1=E, 2=I
        rf_locations: [n_neurons] RF locations (0 for I units)
        sparsity_p: Probability of receiving each feature factor
        seed: Random seed for reproducibility
        experiment_type: 'E2', 'E3a', 'E3b', or 'E4'

    Returns:
        InputAssignment for primary test condition
    """
    rng = np.random.RandomState(seed)

    # Spatial factors: only E units with valid RF
    is_e = neuron_type == 1
    receives_spatial = is_e.copy()

    # Feature factors: sparse random (independent for each neuron and factor)
    receives_salience = rng.rand(n_neurons) < sparsity_p
    receives_identity = rng.rand(n_neurons) < sparsity_p

    return InputAssignment(
        receives_reward=receives_spatial,
        receives_probability=receives_spatial,
        spatial_from_trial_start=True,
        receives_salience=receives_salience,
        receives_identity=receives_identity,
        rf_locations=rf_locations,
        experiment_type=experiment_type,
        sparsity_p=sparsity_p,
        seed=seed
    )


def create_assignment_for_experiment(
    experiment_type: str,
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: np.ndarray,
    seed: int = 0
) -> InputAssignment:
    """
    Factory function to create assignment for any experiment type.

    Args:
        experiment_type: 'E0', 'E1', 'E2', 'E3a', 'E3b', or 'E4'
        n_neurons: Total number of neurons
        neuron_type: [n_neurons] array, 1=E, 2=I
        rf_locations: [n_neurons] RF locations (0 for I units)
        seed: Random seed for sparse assignment

    Returns:
        InputAssignment configured for the experiment
    """
    if experiment_type == 'E0':
        return create_baseline_assignment(n_neurons, neuron_type, rf_locations)
    elif experiment_type == 'E1':
        return create_temporal_fix_assignment(n_neurons, neuron_type, rf_locations)
    elif experiment_type == 'E2':
        return create_primary_assignment(
            n_neurons, neuron_type, rf_locations,
            sparsity_p=0.5, seed=seed, experiment_type='E2'
        )
    elif experiment_type == 'E3a':
        return create_primary_assignment(
            n_neurons, neuron_type, rf_locations,
            sparsity_p=0.25, seed=seed, experiment_type='E3a'
        )
    elif experiment_type == 'E3b':
        return create_primary_assignment(
            n_neurons, neuron_type, rf_locations,
            sparsity_p=0.75, seed=seed, experiment_type='E3b'
        )
    elif experiment_type == 'E4':
        return create_primary_assignment(
            n_neurons, neuron_type, rf_locations,
            sparsity_p=0.5, seed=seed, experiment_type='E4'
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


# =============================================================================
# Biological Input Assignment (Based on SC Afferent Anatomy)
# =============================================================================

def create_biological_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: np.ndarray,
    e_fractions: Dict[str, float] = None,
    i_fractions: Dict[str, float] = None,
    seed: int = 0,
    experiment_type: str = 'BIO'
) -> InputAssignment:
    """
    Create input assignment based on SC afferent anatomy.

    Anatomical basis:
    - SNr (substantia nigra pars reticulata): ~28% E, ~39% I receive value signals
    - V1 (primary visual cortex): 67-85% E, 11-33% I receive feature signals

    Default fractions:
    - Reward/Probability (SNr-like): 30% E, 40% I
    - Salience/Identity (V1-like): 75% E, 25% I

    Args:
        n_neurons: Total number of neurons
        neuron_type: [n_neurons] array, 1=E, 2=I
        rf_locations: [n_neurons] RF locations (0 for I units)
        e_fractions: Dict of factor->fraction for E neurons
                    e.g., {'reward': 0.3, 'probability': 0.3, 'salience': 0.75, 'identity': 0.75}
        i_fractions: Dict of factor->fraction for I neurons
                    e.g., {'reward': 0.4, 'probability': 0.4, 'salience': 0.25, 'identity': 0.25}
        seed: Random seed for reproducibility
        experiment_type: Identifier for experiment

    Returns:
        InputAssignment with biologically-motivated fractions
    """
    # Default fractions based on anatomy
    if e_fractions is None:
        e_fractions = {
            'reward': 0.30,       # SNr-like pathway
            'probability': 0.30,  # SNr-like pathway
            'salience': 0.75,     # V1-like pathway
            'identity': 0.75      # V1-like pathway
        }
    if i_fractions is None:
        i_fractions = {
            'reward': 0.40,       # SNr-like pathway (slight I preference)
            'probability': 0.40,  # SNr-like pathway
            'salience': 0.25,     # V1-like pathway (E preference)
            'identity': 0.25      # V1-like pathway
        }

    rng = np.random.RandomState(seed)

    # Masks for E and I neurons
    is_e = neuron_type == 1
    is_i = neuron_type == 2
    n_e = is_e.sum()
    n_i = is_i.sum()

    # Allocate arrays
    receives_reward = np.zeros(n_neurons, dtype=bool)
    receives_probability = np.zeros(n_neurons, dtype=bool)
    receives_salience = np.zeros(n_neurons, dtype=bool)
    receives_identity = np.zeros(n_neurons, dtype=bool)

    # Assign reward
    e_indices = np.where(is_e)[0]
    i_indices = np.where(is_i)[0]

    # Shuffle and select fraction for each factor and cell type
    for factor, e_frac, i_frac, receives_arr in [
        ('reward', e_fractions.get('reward', 0.3), i_fractions.get('reward', 0.4), receives_reward),
        ('probability', e_fractions.get('probability', 0.3), i_fractions.get('probability', 0.4), receives_probability),
        ('salience', e_fractions.get('salience', 0.75), i_fractions.get('salience', 0.25), receives_salience),
        ('identity', e_fractions.get('identity', 0.75), i_fractions.get('identity', 0.25), receives_identity),
    ]:
        # E neurons
        n_e_receive = int(np.round(n_e * e_frac))
        e_shuffled = rng.permutation(e_indices)
        receives_arr[e_shuffled[:n_e_receive]] = True

        # I neurons
        n_i_receive = int(np.round(n_i * i_frac))
        i_shuffled = rng.permutation(i_indices)
        receives_arr[i_shuffled[:n_i_receive]] = True

    # Compute effective sparsity (average fraction receiving any factor)
    total_assignments = (receives_reward.sum() + receives_probability.sum() +
                        receives_salience.sum() + receives_identity.sum())
    effective_sparsity = total_assignments / (4 * n_neurons)

    return InputAssignment(
        receives_reward=receives_reward,
        receives_probability=receives_probability,
        spatial_from_trial_start=True,  # Spatial factors from trial start
        receives_salience=receives_salience,
        receives_identity=receives_identity,
        rf_locations=rf_locations,
        experiment_type=experiment_type,
        sparsity_p=effective_sparsity,  # Store effective sparsity
        seed=seed
    )


def create_extreme_biological_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: np.ndarray,
    seed: int = 0
) -> InputAssignment:
    """
    Create extreme biological assignment for sensitivity testing.

    Extreme constraint:
    - E neurons: 100% receive features (salience/identity), 0% receive values (reward/prob)
    - I neurons: 100% receive values, 0% receive features

    This tests whether the model can still fit with complete pathway separation.
    """
    return create_biological_assignment(
        n_neurons=n_neurons,
        neuron_type=neuron_type,
        rf_locations=rf_locations,
        e_fractions={'reward': 0.0, 'probability': 0.0, 'salience': 1.0, 'identity': 1.0},
        i_fractions={'reward': 1.0, 'probability': 1.0, 'salience': 0.0, 'identity': 0.0},
        seed=seed,
        experiment_type='BIO_EXTREME'
    )


def save_assignment(assignment: InputAssignment, filepath: str) -> None:
    """Save assignment to file."""
    np.save(filepath, assignment.to_dict(), allow_pickle=True)


def load_assignment(filepath: str) -> InputAssignment:
    """Load assignment from file."""
    d = np.load(filepath, allow_pickle=True).item()
    return InputAssignment.from_dict(d)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_assignment_statistics(assignment: InputAssignment) -> Dict[str, Any]:
    """
    Compute statistics about an assignment for reporting.

    Returns:
        Dictionary with counts and fractions for each factor
    """
    n_neurons = len(assignment.receives_reward)
    n_e = (assignment.rf_locations > 0).sum()
    n_i = n_neurons - n_e

    stats = {
        'n_neurons': n_neurons,
        'n_e': int(n_e),
        'n_i': int(n_i),
        'experiment_type': assignment.experiment_type,
        'sparsity_p': assignment.sparsity_p,
        'seed': assignment.seed,
        'spatial_from_trial_start': assignment.spatial_from_trial_start,
    }

    for factor in ['reward', 'probability', 'salience', 'identity']:
        receives = assignment.get_input_matrix(factor)
        stats[f'{factor}_n_receivers'] = int(receives.sum())
        stats[f'{factor}_frac_receivers'] = float(receives.sum() / n_neurons)

        # E vs I breakdown
        e_mask = assignment.rf_locations > 0
        stats[f'{factor}_n_e_receivers'] = int(receives[e_mask].sum())
        stats[f'{factor}_n_i_receivers'] = int(receives[~e_mask].sum())

    return stats


if __name__ == "__main__":
    # Quick test
    print("Testing input assignment module...")

    n = 50
    neuron_type = np.array([1] * 40 + [2] * 10)  # 40 E, 10 I
    rf_locations = np.zeros(n, dtype=int)
    for i in range(40):
        rf_locations[i] = (i % 4) + 1

    print("\n--- E0 Baseline ---")
    e0 = create_baseline_assignment(n, neuron_type, rf_locations)
    print(e0.summary())

    print("\n--- E1 Temporal Fix ---")
    e1 = create_temporal_fix_assignment(n, neuron_type, rf_locations)
    print(e1.summary())

    print("\n--- E2 Primary (p=0.5, seed=0) ---")
    e2 = create_primary_assignment(n, neuron_type, rf_locations, sparsity_p=0.5, seed=0)
    print(e2.summary())

    print("\n--- E2 Primary (p=0.5, seed=1) ---")
    e2_s1 = create_primary_assignment(n, neuron_type, rf_locations, sparsity_p=0.5, seed=1)
    print(e2_s1.summary())

    print("\n--- E3a Sparse (p=0.25) ---")
    e3a = create_primary_assignment(n, neuron_type, rf_locations, sparsity_p=0.25, seed=0)
    print(e3a.summary())

    # Test serialization
    print("\n--- Serialization Test ---")
    d = e2.to_dict()
    e2_loaded = InputAssignment.from_dict(d)
    assert np.allclose(e2.receives_salience, e2_loaded.receives_salience)
    print("Serialization OK")

    # Test statistics
    print("\n--- Statistics ---")
    stats = compute_assignment_statistics(e2)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nAll tests passed!")
