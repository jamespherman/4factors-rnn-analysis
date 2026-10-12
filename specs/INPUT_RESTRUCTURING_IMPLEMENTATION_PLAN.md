# Input Restructuring Implementation Plan

## Overview

This document details the implementation plan for the input restructuring experiments defined in `INPUT_RESTRUCTURING_SPEC.md`. The goal is to test whether SC factor selectivity arises from recurrent dynamics (involving interneurons) or direct input drive.

## Phase 1: RF Inference for E Units

### 1.1 Function: `infer_rf_locations()`

**Location:** `src/data_loader.py` (add new function)

**Approach:**
1. Load the raw data with trial-level information
2. For each neuron classified as E (classic SC, neuron_type == 1):
   - Group trials by target location (1-4)
   - Compute mean firing rate in visual epoch (50-250ms post-target onset) for each location
   - Assign RF = argmax location
3. For I units (neuron_type == 2): return `None` (no RF assignment)

**Implementation:**
```python
def infer_rf_locations(data: dict, visual_epoch_ms: Tuple[int, int] = (50, 250)) -> np.ndarray:
    """
    Infer receptive field location for each neuron based on spatial response profile.

    Args:
        data: Loaded session data dictionary
        visual_epoch_ms: Start and end of visual epoch relative to target onset

    Returns:
        rf_locations: [n_neurons] array of location indices (1-4), or 0 for I units
    """
    n_neurons = int(data['n_neurons'])
    n_trials = int(data['n_trials'])
    bin_size_ms = float(data['bin_size_ms'])

    firing_rates = data['firing_rates']  # [n_neurons, n_bins, n_trials]
    neuron_type = data['neuron_type']    # [n_neurons], 1=E, 2=I
    trial_location = data['trial_location']  # [n_trials], 1-4

    # Find target onset bin from input_target_loc
    target_loc = data['input_target_loc']  # [4, n_bins, n_trials]
    target_onset_bin = find_target_onset_bin(target_loc)

    # Visual epoch bins
    epoch_start = target_onset_bin + int(visual_epoch_ms[0] / bin_size_ms)
    epoch_end = target_onset_bin + int(visual_epoch_ms[1] / bin_size_ms)

    rf_locations = np.zeros(n_neurons, dtype=int)

    for i in range(n_neurons):
        if neuron_type[i] == 2:  # I unit - no RF
            rf_locations[i] = 0
            continue

        # E unit - compute mean response per location
        location_responses = np.zeros(4)
        for loc in range(1, 5):
            loc_trials = trial_location == loc
            if loc_trials.sum() > 0:
                loc_fr = firing_rates[i, epoch_start:epoch_end, loc_trials]
                location_responses[loc-1] = np.nanmean(loc_fr)

        # RF = location with maximum response
        rf_locations[i] = np.argmax(location_responses) + 1  # 1-indexed

    return rf_locations
```

### 1.2 Helper: `find_target_onset_bin()`

```python
def find_target_onset_bin(input_target_loc: np.ndarray) -> int:
    """Find the first time bin where target appears (any location)."""
    # input_target_loc: [4, n_bins, n_trials]
    target_any = input_target_loc.sum(axis=0)  # [n_bins, n_trials]
    first_onset_per_trial = (target_any > 0).argmax(axis=0)
    return int(np.median(first_onset_per_trial))
```

---

## Phase 2: Input Assignment System

### 2.1 New Module: `src/input_assignment.py`

This module handles the creation and storage of input assignments for each experiment type.

**Core Data Structure:**
```python
@dataclass
class InputAssignment:
    """Tracks which neurons receive which input factors."""

    # Spatial factors (reward, probability)
    receives_reward: np.ndarray      # [n_neurons] bool - RF-matched for E, False for I
    receives_probability: np.ndarray # [n_neurons] bool

    # Temporal structure
    spatial_from_trial_start: bool   # True = from t=0, False = from target onset

    # Feature factors (salience, identity) - sparse random
    receives_salience: np.ndarray    # [n_neurons] bool
    receives_identity: np.ndarray    # [n_neurons] bool

    # RF information (for E units)
    rf_locations: np.ndarray         # [n_neurons] int, 0 for I units

    # Metadata
    experiment_type: str             # 'E0', 'E1', 'E2', etc.
    sparsity_p: float               # Probability of receiving feature factor
    seed: int                       # Random seed for sparse assignment

    def to_dict(self) -> dict:
        """Serialize for saving."""
        pass

    @classmethod
    def from_dict(cls, d: dict) -> 'InputAssignment':
        """Load from saved dict."""
        pass
```

**Factory Functions:**
```python
def create_baseline_assignment(n_neurons: int, neuron_type: np.ndarray) -> InputAssignment:
    """E0: All neurons receive all factors at target onset."""
    return InputAssignment(
        receives_reward=np.ones(n_neurons, dtype=bool),
        receives_probability=np.ones(n_neurons, dtype=bool),
        spatial_from_trial_start=False,  # At target onset
        receives_salience=np.ones(n_neurons, dtype=bool),
        receives_identity=np.ones(n_neurons, dtype=bool),
        rf_locations=np.arange(1, 5)[neuron_type == 1],  # Placeholder
        experiment_type='E0',
        sparsity_p=1.0,
        seed=0
    )

def create_temporal_fix_assignment(n_neurons: int, neuron_type: np.ndarray) -> InputAssignment:
    """E1: All neurons, but spatial from trial start."""
    return InputAssignment(
        receives_reward=np.ones(n_neurons, dtype=bool),
        receives_probability=np.ones(n_neurons, dtype=bool),
        spatial_from_trial_start=True,  # From trial start
        receives_salience=np.ones(n_neurons, dtype=bool),
        receives_identity=np.ones(n_neurons, dtype=bool),
        rf_locations=np.arange(1, 5)[neuron_type == 1],
        experiment_type='E1',
        sparsity_p=1.0,
        seed=0
    )

def create_primary_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: np.ndarray,
    sparsity_p: float = 0.5,
    seed: int = 0
) -> InputAssignment:
    """E2/E3/E4: RF-matched spatial for E, sparse feature factors."""
    rng = np.random.RandomState(seed)

    # Spatial: only E units with RF receive their RF-matched values
    is_e = neuron_type == 1
    receives_spatial = is_e.copy()  # I units get nothing

    # Feature factors: sparse random
    receives_salience = rng.rand(n_neurons) < sparsity_p
    receives_identity = rng.rand(n_neurons) < sparsity_p

    return InputAssignment(
        receives_reward=receives_spatial,
        receives_probability=receives_spatial,
        spatial_from_trial_start=True,
        receives_salience=receives_salience,
        receives_identity=receives_identity,
        rf_locations=rf_locations,
        experiment_type='E2',
        sparsity_p=sparsity_p,
        seed=seed
    )
```

---

## Phase 3: Modified Input Construction

### 3.1 New Function: `construct_restructured_input_tensor()`

**Location:** `src/data_loader.py`

This function constructs inputs according to the specified assignment scheme.

**New Input Channel Layout (variable per assignment):**
- For E units with RF at location k:
  - `reward_at_rf`: Scalar value of reward at that location
  - `prob_at_rf`: Scalar value of probability at that location
  - `target_in_rf`: Binary, 1 when target appears at RF location
- For I units:
  - `target_onset`: Global binary signal (target appeared anywhere)
- Feature factors (sparse):
  - `salience`: If assigned, value at target onset; else 0
  - `identity`: If assigned, value at target onset; else 0

**Implementation Approach:**

```python
def construct_restructured_input_tensor(
    data: dict,
    assignment: InputAssignment
) -> np.ndarray:
    """
    Construct neuron-specific input tensor based on assignment.

    Args:
        data: Loaded session data
        assignment: InputAssignment specifying what each neuron receives

    Returns:
        inputs: [n_trials, n_bins, n_neurons, n_input_channels] tensor

    Note: n_input_channels is fixed at 6:
        0: reward_signal (RF-matched for E, 0 for I)
        1: probability_signal (RF-matched for E, 0 for I)
        2: target_signal (target_in_RF for E, global onset for I)
        3: salience (if assigned)
        4: identity (if assigned)
        5: fixation_on (same for all)
    """
    n_trials = int(data['n_trials'])
    n_bins = int(data['n_time_bins'])
    n_neurons = int(data['n_neurons'])

    # Initialize output
    inputs = np.zeros((n_trials, n_bins, n_neurons, 6), dtype=np.float32)

    # Extract raw signals
    trial_reward = data['trial_reward']     # [n_trials] 0/1
    trial_probability = data.get('trial_probability', np.zeros(n_trials))
    trial_location = data['trial_location'] # [n_trials] 1-4
    trial_salience = data.get('trial_salience', np.zeros(n_trials))
    trial_identity = data.get('trial_identity', np.zeros(n_trials))

    fixation_on = data['input_fixation_on'].T  # [n_trials, n_bins]
    target_loc = data['input_target_loc']      # [4, n_bins, n_trials]

    # Find target onset bin
    target_onset_bin = find_target_onset_bin(target_loc)

    for i in range(n_neurons):
        rf = assignment.rf_locations[i]
        is_e = rf > 0  # E units have RF > 0

        for trial in range(n_trials):
            loc = trial_location[trial]

            # Channel 0: Reward signal
            if assignment.receives_reward[i]:
                if is_e:
                    # E unit: reward value for their RF location
                    # Note: trial_reward tells us if reward is high at target location
                    # We need to encode whether high reward is at THIS neuron's RF
                    reward_val = trial_reward[trial] if loc == rf else 0
                else:
                    # I unit: could receive something, but in primary design they don't
                    reward_val = 0

                if assignment.spatial_from_trial_start:
                    inputs[trial, :, i, 0] = reward_val
                else:
                    inputs[trial, target_onset_bin:, i, 0] = reward_val

            # Channel 1: Probability signal (similar logic)
            if assignment.receives_probability[i]:
                prob_val = trial_probability[trial] if (is_e and loc == rf) else 0
                if assignment.spatial_from_trial_start:
                    inputs[trial, :, i, 1] = prob_val
                else:
                    inputs[trial, target_onset_bin:, i, 1] = prob_val

            # Channel 2: Target signal
            if is_e:
                # E unit: target_in_RF
                target_in_rf = 1.0 if loc == rf else 0.0
                inputs[trial, target_onset_bin:, i, 2] = target_in_rf
            else:
                # I unit: global target onset
                inputs[trial, target_onset_bin:, i, 2] = 1.0

            # Channel 3: Salience
            if assignment.receives_salience[i]:
                sal_val = trial_salience[trial]
                inputs[trial, target_onset_bin:, i, 3] = sal_val

            # Channel 4: Identity
            if assignment.receives_identity[i]:
                id_val = trial_identity[trial]
                inputs[trial, target_onset_bin:, i, 4] = id_val

            # Channel 5: Fixation (global, same for all)
            inputs[trial, :, i, 5] = fixation_on[trial]

    return inputs
```

### 3.2 Alternative: Per-Neuron Input Weights

**Consideration:** The above creates a 4D tensor which may be memory-intensive. An alternative is to keep the global input tensor (14 channels) and use a masking matrix applied through `W_in`:

```python
def create_input_mask(assignment: InputAssignment, n_input_channels: int) -> np.ndarray:
    """
    Create [n_neurons, n_input_channels] mask for W_in.

    Elements are 1 if neuron receives that input channel, 0 otherwise.
    Applied as: effective_W_in = W_in * input_mask
    """
    pass
```

**Trade-off:** The 4D tensor approach is conceptually cleaner and makes the analysis straightforward, but the masking approach integrates better with the existing model architecture.

**Decision:** Use the masking approach for computational efficiency, but store the full assignment for analysis.

---

## Phase 4: Modified Model for Neuron-Specific Inputs

### 4.1 Extend EIRNN to Support Input Assignment

**Location:** `src/model.py` - add to EIRNN class

```python
class EIRNN(nn.Module):
    def __init__(
        self,
        # ... existing args ...
        input_assignment: Optional[InputAssignment] = None
    ):
        # ... existing init ...

        if input_assignment is not None:
            self.register_input_assignment(input_assignment)

    def register_input_assignment(self, assignment: InputAssignment):
        """Register input assignment and create appropriate masks."""
        self.input_assignment = assignment

        # Create input mask: [n_total, n_input_channels]
        # This modulates W_in to implement neuron-specific input routing
        input_mask = self._create_input_mask(assignment)
        self.register_buffer('input_mask', torch.tensor(input_mask, dtype=torch.float32))

    def _create_input_mask(self, assignment: InputAssignment) -> np.ndarray:
        """Create mask from assignment specification."""
        # Map assignment to input channel indices
        # Return [n_total, n_inputs] binary mask
        pass
```

### 4.2 Modified Forward Pass

The model's forward pass needs to apply the input mask to implement neuron-specific routing:

```python
def forward(self, inputs, ...):
    # ...
    for t in range(n_steps):
        # External input with masking
        if hasattr(self, 'input_mask') and self.input_mask is not None:
            # Apply mask: each neuron only receives inputs it's assigned
            ext_input = torch.matmul(inputs[:, t, :], (self.W_in * self.input_mask).T)
        else:
            ext_input = torch.matmul(inputs[:, t, :], self.W_in.T)
        # ... rest of forward pass ...
```

---

## Phase 5: Emergent Selectivity Analysis

### 5.1 New Module: `src/analysis/emergent_selectivity.py`

```python
def compute_factor_selectivity(
    model_rates: np.ndarray,      # [n_trials, n_bins, n_neurons]
    factor_values: np.ndarray,    # [n_trials] factor value per trial
    epoch_bins: Tuple[int, int]   # Time bins to analyze
) -> np.ndarray:
    """
    Compute selectivity for a given factor for each neuron.

    Returns:
        selectivity: [n_neurons] Pearson correlation between epoch-averaged
                     firing rate and factor value
    """
    n_neurons = model_rates.shape[2]

    # Epoch-averaged rates
    epoch_rates = model_rates[:, epoch_bins[0]:epoch_bins[1], :].mean(axis=1)  # [n_trials, n_neurons]

    selectivity = np.zeros(n_neurons)
    for i in range(n_neurons):
        r = np.corrcoef(epoch_rates[:, i], factor_values)[0, 1]
        selectivity[i] = r if not np.isnan(r) else 0.0

    return selectivity


def compute_emergent_selectivity(
    model_rates: np.ndarray,
    factor_values: np.ndarray,
    assignment: InputAssignment,
    factor_name: str,           # 'reward', 'probability', 'salience', 'identity'
    epoch_bins: Tuple[int, int]
) -> dict:
    """
    Compute emergent selectivity: selectivity for factors the neuron doesn't receive.

    Returns:
        dict with:
            - selectivity: [n_neurons] all selectivity values
            - receives_input: [n_neurons] bool
            - emergent_selectivity: [n_neurons] selectivity for non-receivers only
            - stats: summary statistics
    """
    # Get which neurons receive this factor
    if factor_name == 'reward':
        receives_input = assignment.receives_reward
    elif factor_name == 'probability':
        receives_input = assignment.receives_probability
    elif factor_name == 'salience':
        receives_input = assignment.receives_salience
    elif factor_name == 'identity':
        receives_input = assignment.receives_identity
    else:
        raise ValueError(f"Unknown factor: {factor_name}")

    # Compute selectivity for all neurons
    selectivity = compute_factor_selectivity(model_rates, factor_values, epoch_bins)

    # Emergent selectivity: only for neurons that don't receive this factor
    emergent_mask = ~receives_input
    emergent_selectivity = selectivity.copy()
    emergent_selectivity[receives_input] = np.nan  # Mask direct receivers

    # Statistics
    n_emergent = emergent_mask.sum()
    if n_emergent > 0:
        emergent_vals = selectivity[emergent_mask]
        significant_threshold = 0.2  # Arbitrary; could use permutation test
        n_significant = (np.abs(emergent_vals) > significant_threshold).sum()
        mean_emergent = np.mean(np.abs(emergent_vals))
    else:
        n_significant = 0
        mean_emergent = 0.0

    return {
        'selectivity': selectivity,
        'receives_input': receives_input,
        'emergent_selectivity': emergent_selectivity,
        'stats': {
            'n_emergent_neurons': int(n_emergent),
            'n_significant_emergent': int(n_significant),
            'mean_abs_emergent_selectivity': float(mean_emergent),
            'fraction_significant': float(n_significant / n_emergent) if n_emergent > 0 else 0.0
        }
    }
```

---

## Phase 6: Recurrence Ablation

### 6.1 New Module: `src/analysis/ablation.py`

```python
import copy
import torch
import numpy as np

def create_ablated_model(model, ablation_type: str = 'full_recurrence'):
    """
    Create a copy of the model with specified ablation.

    Args:
        model: Trained EIRNN
        ablation_type:
            - 'full_recurrence': Zero all W_rec
            - 'ie_only': Zero only I->E connections
            - 'ei_only': Zero only E->I connections

    Returns:
        ablated_model: Model with ablated weights
    """
    ablated_model = copy.deepcopy(model)

    with torch.no_grad():
        if ablation_type == 'full_recurrence':
            ablated_model.W_rec_raw.zero_()
        elif ablation_type == 'ie_only':
            # I->E: rows 0:n_exc, cols n_exc:n_total
            ablated_model.W_rec_raw[:model.n_exc, model.n_exc:] = 0
        elif ablation_type == 'ei_only':
            # E->I: rows n_exc:n_total, cols 0:n_exc
            ablated_model.W_rec_raw[model.n_exc:, :model.n_exc] = 0
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

    return ablated_model


def run_ablation_analysis(
    model,
    inputs: torch.Tensor,
    factor_values: dict,          # {factor_name: [n_trials] values}
    assignment: InputAssignment,
    epoch_bins: Tuple[int, int],
    ablation_types: List[str] = ['full_recurrence', 'ie_only']
) -> dict:
    """
    Run ablation analysis comparing selectivity before/after ablation.

    Returns:
        dict with selectivity changes for each factor and ablation type
    """
    results = {}

    # Original model selectivity
    model.eval()
    with torch.no_grad():
        rates_original, _ = model(inputs)
        rates_original = rates_original.cpu().numpy()

    for factor_name, factor_vals in factor_values.items():
        results[factor_name] = {}

        # Original selectivity
        orig_results = compute_emergent_selectivity(
            rates_original, factor_vals, assignment, factor_name, epoch_bins
        )
        results[factor_name]['original'] = orig_results

        # Ablated selectivity
        for abl_type in ablation_types:
            ablated_model = create_ablated_model(model, abl_type)
            ablated_model.eval()

            with torch.no_grad():
                rates_ablated, _ = ablated_model(inputs)
                rates_ablated = rates_ablated.cpu().numpy()

            abl_results = compute_emergent_selectivity(
                rates_ablated, factor_vals, assignment, factor_name, epoch_bins
            )
            results[factor_name][abl_type] = abl_results

            # Compute change in emergent selectivity
            emergent_mask = ~assignment.receives_input[factor_name]
            if emergent_mask.sum() > 0:
                orig_sel = np.abs(orig_results['selectivity'][emergent_mask]).mean()
                abl_sel = np.abs(abl_results['selectivity'][emergent_mask]).mean()
                results[factor_name][f'{abl_type}_change'] = {
                    'original_mean': float(orig_sel),
                    'ablated_mean': float(abl_sel),
                    'reduction_fraction': float(1 - abl_sel / orig_sel) if orig_sel > 0 else 0
                }

    return results
```

---

## Phase 7: Experimental Runner Script

### 7.1 New Script: `scripts/run_input_restructuring.py`

```python
#!/usr/bin/env python3
"""
Run the input restructuring experimental matrix.

Usage:
    python scripts/run_input_restructuring.py                    # Full run
    python scripts/run_input_restructuring.py --priority-only    # E0, E1, E2_seed0 only
    python scripts/run_input_restructuring.py --experiment E2    # Specific experiment
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# ... imports ...

# Experimental matrix
EXPERIMENTS = {
    'E0': {
        'description': 'Baseline: all neurons, all factors, at target onset',
        'assignment_fn': 'baseline',
        'sparsity_p': 1.0,
        'seeds': [0],
        'spatial_from_start': False
    },
    'E1': {
        'description': 'Temporal fix: all neurons, spatial from trial start',
        'assignment_fn': 'temporal_fix',
        'sparsity_p': 1.0,
        'seeds': [0],
        'spatial_from_start': True
    },
    'E2': {
        'description': 'Primary test: RF-matched spatial, sparse features (p=0.5)',
        'assignment_fn': 'primary',
        'sparsity_p': 0.5,
        'seeds': [0, 1, 2, 3, 4],
        'spatial_from_start': True
    },
    'E3a': {
        'description': 'Sparsity sensitivity: p=0.25',
        'assignment_fn': 'primary',
        'sparsity_p': 0.25,
        'seeds': [0, 1, 2],
        'spatial_from_start': True
    },
    'E3b': {
        'description': 'Sparsity sensitivity: p=0.75',
        'assignment_fn': 'primary',
        'sparsity_p': 0.75,
        'seeds': [0, 1, 2],
        'spatial_from_start': True
    },
    'E4': {
        'description': 'Replication with different seeds',
        'assignment_fn': 'primary',
        'sparsity_p': 0.5,
        'seeds': [100, 101, 102, 103, 104],  # Different seed range
        'spatial_from_start': True
    }
}

def run_single_experiment(
    exp_name: str,
    seed: int,
    data: dict,
    output_dir: Path,
    config: dict
) -> dict:
    """Run a single experiment configuration."""

    exp_config = EXPERIMENTS[exp_name]
    run_name = f"{exp_name}_seed{seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {run_name}")
    print(f"  {exp_config['description']}")
    print(f"{'='*60}")

    # 1. Infer RF locations
    rf_locations = infer_rf_locations(data)

    # 2. Create input assignment
    n_neurons = int(data['n_neurons'])
    neuron_type = data['neuron_type']

    if exp_config['assignment_fn'] == 'baseline':
        assignment = create_baseline_assignment(n_neurons, neuron_type)
    elif exp_config['assignment_fn'] == 'temporal_fix':
        assignment = create_temporal_fix_assignment(n_neurons, neuron_type)
    elif exp_config['assignment_fn'] == 'primary':
        assignment = create_primary_assignment(
            n_neurons, neuron_type, rf_locations,
            sparsity_p=exp_config['sparsity_p'],
            seed=seed
        )

    # Save assignment
    np.save(run_dir / 'input_assignment.npy', assignment.to_dict())

    # 3. Construct inputs
    # ... (use modified data loader with assignment)

    # 4. Create and train model
    model = create_model_from_data(...)

    # Training loop (adapted from train_final_model.py)
    # ...

    # 5. Compute selectivity analysis
    # ...

    # 6. Save results
    # ...

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--priority-only', action='store_true',
                       help='Only run E0, E1, E2_seed0')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Run specific experiment (E0, E1, E2, E3a, E3b, E4)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Run specific seed only')
    parser.add_argument('--data-path', type=str,
                       default='data/rnn_export_Newton_08_15_2025_SC.mat')
    parser.add_argument('--output-dir', type=str,
                       default='results/input_restructuring')
    args = parser.parse_args()

    # Determine which experiments to run
    if args.priority_only:
        experiments_to_run = [
            ('E0', 0),
            ('E1', 0),
            ('E2', 0)
        ]
    elif args.experiment:
        exp = EXPERIMENTS[args.experiment]
        seeds = [args.seed] if args.seed is not None else exp['seeds']
        experiments_to_run = [(args.experiment, s) for s in seeds]
    else:
        # Full matrix
        experiments_to_run = []
        for exp_name, exp_config in EXPERIMENTS.items():
            for seed in exp_config['seeds']:
                experiments_to_run.append((exp_name, seed))

    print(f"Running {len(experiments_to_run)} experiments")

    # Load data once
    data = load_mat_file(args.data_path)

    # Run experiments
    all_results = {}
    for exp_name, seed in experiments_to_run:
        try:
            results = run_single_experiment(exp_name, seed, data,
                                           Path(args.output_dir), {})
            all_results[f"{exp_name}_seed{seed}"] = results
        except Exception as e:
            print(f"ERROR in {exp_name}_seed{seed}: {e}")
            all_results[f"{exp_name}_seed{seed}"] = {'error': str(e)}

    # Save summary
    with open(Path(args.output_dir) / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experiments complete. Results in {args.output_dir}")


if __name__ == '__main__':
    main()
```

---

## Phase 8: Post-Training Analysis and Figures

### 8.1 Analysis Script: `scripts/analyze_input_restructuring.py`

After all experiments complete, this script:

1. **Aggregates metrics across experiments**
2. **Computes emergent selectivity summary**
3. **Analyzes I→E weight structure**
4. **Runs ablation on E2 models**
5. **Assesses cross-seed consistency**
6. **Generates publication figures**

### 8.2 Key Figures

1. **Fit Quality Comparison** (`fit_quality_comparison.png`)
   - Bar chart: PSTH correlation for E0, E1, E2 (mean±SE across seeds)
   - Horizontal line at 80% of E0 as success threshold

2. **Emergent Selectivity by Experiment** (`emergent_selectivity_by_experiment.png`)
   - For each factor (reward, prob, salience, identity):
     - Bar for each experiment showing fraction of emergent neurons with significant selectivity
     - Error bars for seeded experiments

3. **I→E Weight Clustering** (`ie_weight_clustering.png`)
   - Heatmap of I→E weights sorted by factor selectivity
   - Side panel showing I unit selectivity

4. **Ablation Results** (`ablation_results.png`)
   - Before/after bar plot for each factor
   - Shows reduction in emergent selectivity

5. **Cross-Seed Consistency** (`cross_seed_consistency.png`)
   - Correlation matrix of I→E weight structure across E2 seeds

---

## File Structure Summary

```
src/
├── data_loader.py           # Modified: add infer_rf_locations, construct_restructured_input_tensor
├── input_assignment.py      # NEW: InputAssignment class, factory functions
├── model.py                 # Modified: support input_assignment, input_mask
├── analysis/
│   ├── __init__.py
│   ├── emergent_selectivity.py  # NEW
│   └── ablation.py              # NEW

scripts/
├── run_input_restructuring.py   # NEW: main experiment runner
└── analyze_input_restructuring.py  # NEW: post-training analysis

results/input_restructuring/
├── E0_seed0/
│   ├── model_best.pt
│   ├── input_assignment.npy
│   ├── metrics.json
│   └── ...
├── E1_seed0/
├── E2_seed0/
├── E2_seed1/
├── ...
├── analysis/
│   ├── emergent_selectivity_summary.csv
│   ├── ie_weight_structure.csv
│   ├── cross_seed_consistency.csv
│   └── figures/
└── REPORT.md
```

---

## Implementation Order

1. **Day 1: Core Infrastructure**
   - [ ] Implement `infer_rf_locations()` in data_loader.py
   - [ ] Create `src/input_assignment.py` with InputAssignment class
   - [ ] Implement factory functions for E0, E1, E2 assignments

2. **Day 2: Input Construction**
   - [ ] Implement `construct_restructured_input_tensor()` or masking approach
   - [ ] Modify EIRNN to support input_assignment
   - [ ] Test with synthetic data

3. **Day 3: Analysis Modules**
   - [ ] Create `src/analysis/emergent_selectivity.py`
   - [ ] Create `src/analysis/ablation.py`
   - [ ] Unit tests for analysis functions

4. **Day 4: Experiment Runner**
   - [ ] Create `scripts/run_input_restructuring.py`
   - [ ] Run priority experiments (E0, E1, E2_seed0)
   - [ ] Verify fit quality maintained

5. **Day 5-6: Full Experimental Matrix**
   - [ ] Run remaining E2 seeds
   - [ ] Run E3a, E3b experiments
   - [ ] Run E4 replication

6. **Day 7: Analysis and Reporting**
   - [ ] Create `scripts/analyze_input_restructuring.py`
   - [ ] Generate figures
   - [ ] Write `results/input_restructuring/REPORT.md`

---

## Risk Mitigation

1. **Fit Quality Degradation**
   - Monitor PSTH correlation during training
   - If E2 drops below 60% of E0, consider:
     - Increasing hidden units
     - Longer training
     - Relaxing sparsity constraint

2. **Memory Issues with 4D Tensor**
   - Use masking approach instead
   - Or process trials in smaller batches

3. **Long Training Times**
   - Use GPU if available
   - Save checkpoints frequently
   - Priority ordering ensures key results first

4. **Statistical Power for Emergent Selectivity**
   - With 5 seeds, may have limited power
   - Focus on effect sizes, not just significance
   - Consider bootstrap confidence intervals

---

## Success Metrics (from INPUT_RESTRUCTURING_SPEC.md)

1. **Fit quality preserved:** E2 achieves ≥80% of E0's validation correlation
2. **Emergent selectivity detected:** Significant factor selectivity in neurons lacking direct input
3. **Recurrence dependence:** Ablation reduces emergent selectivity by ≥50%
4. **Cross-seed consistency:** I→E weight structure correlates r≥0.5 across seeds
5. **I unit involvement:** I→E weights show non-random structure w.r.t. target E selectivity
