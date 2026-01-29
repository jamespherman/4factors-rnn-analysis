# INPUT_RESTRUCTURING_SPEC.md — Experimental Plan for Testing Recurrence-Dependent Factor Selectivity

## Scientific Motivation

The core question: Does SC factor selectivity arise from local recurrent dynamics (involving interneurons) or is it driven entirely by input?

**The problem with current input encoding:** If every neuron receives explicit factor labels (reward, salience, probability, identity) tied to target onset, the network has no computational need for recurrence. It can achieve factor selectivity via direct input→output mapping, rendering the I→E weight analysis uninterpretable.

**The solution:** Restructure inputs so that:
1. **Temporal validity:** Reward/probability signals are available from trial start (reflecting learned block contingencies), not target onset
2. **Spatial heterogeneity:** Not all neurons receive all factor information—forcing the network to use recurrence if it needs to integrate across factors

**Key test:** If a neuron shows selectivity for a factor it doesn't directly receive as input, that selectivity *must* arise from recurrent dynamics.

## Input Restructuring Design

### Factor Classification

| Factor | Information type | When monkey knows | Proposed encoding |
|--------|-----------------|-------------------|-------------------|
| **Reward** | Learned spatial association | Block start (persists) | RF-matched, from trial start |
| **Probability** | Learned spatial prior | Block start (persists) | RF-matched, from trial start |
| **Salience** | Stimulus-driven feature | Target onset | Sparse random, at target onset |
| **Identity** | Stimulus-driven feature | Target onset | Sparse random, at target onset |

### RF Assignment for Recorded Neurons

**Classic SC neurons (E units):** Infer RF location from spatial tuning:
- Compute mean visually-evoked firing rate for each target location (collapsed across factors)
- Assign RF = location with maximum response
- Fallback: use saccade-related activity if visual response is weak

**Putative interneurons (I units):** Do NOT assign RF-specific inputs. Rationale:
- These neurons are defined by lacking clear spatial selectivity (bilateral responses or visual suppression)
- If they show factor modulation, it should arise from recurrent input, not direct drive
- This is a strong test: interneurons receive NO spatial factor information directly

### Input Channels (Proposed)

```
Inputs to E units (classic SC neurons):
├── Spatial expectation (from trial start, continuous):
│   ├── reward_at_RF: scalar, reward magnitude for this neuron's RF location
│   └── prob_at_RF: scalar, probability for this neuron's RF location
├── Target onset (step function at target appearance):
│   └── target_in_RF: binary, whether target appeared in this neuron's RF
└── Feature factors (sparse random assignment):
    ├── salience: scalar (if assigned, else 0)
    └── identity: scalar (if assigned, else 0)

Inputs to I units (putative interneurons):
├── Spatial expectation: NONE (must derive from recurrence)
├── Target onset:
│   └── target_onset: binary, global signal that target appeared
└── Feature factors (sparse random assignment):
    ├── salience: scalar (if assigned, else 0)
    └── identity: scalar (if assigned, else 0)
```

### Sparse Random Assignment

For feature factors (salience, identity), each neuron independently receives or doesn't receive each factor:
- `receives_salience[i] ~ Bernoulli(p_sparse)`
- `receives_identity[i] ~ Bernoulli(p_sparse)`
- Assignment is fixed for a given random seed, varies across experiments

## Experimental Matrix

| Experiment | Spatial factors (rew/prob) | Feature factors (sal/id) | Seeds | Purpose |
|------------|---------------------------|-------------------------|-------|---------|
| **E0** | All neurons, at target onset | All neurons, at target onset | 1 | Baseline (current approach) |
| **E1** | All neurons, from trial start | All neurons, at target onset | 1 | Temporal fix only |
| **E2** | RF-matched (E) / None (I), from trial start | Sparse p=0.5, at target onset | 5 | Primary test |
| **E3a** | RF-matched (E) / None (I), from trial start | Sparse p=0.25, at target onset | 3 | Sparsity sensitivity |
| **E3b** | RF-matched (E) / None (I), from trial start | Sparse p=0.75, at target onset | 3 | Sparsity sensitivity |
| **E4** | RF-matched (E) / None (I), from trial start | Sparse p=0.5, at target onset | 5 (different) | Replication with new seeds |

**Total models to train:** 1 + 1 + 5 + 3 + 3 + 5 = 18

## Metrics and Analysis

### 1. Fit Quality
- Validation PSTH correlation (overall and per-neuron)
- Compare to E0 baseline—how much does restructuring hurt fit?

### 2. Emergent Selectivity Analysis
For each neuron and each factor:
- `direct_input[i, f]`: Did neuron i receive factor f as direct input? (binary)
- `selectivity[i, f]`: Correlation between neuron i's activity and factor f value (computed from model outputs)

**Key analysis:** Among neurons where `direct_input[i, f] = False`, how many show significant `selectivity[i, f]`? This is "emergent" selectivity that must arise from recurrence.

### 3. I→E Weight Structure
- Extract `W_rec[I_units, E_units]` from trained models
- For each I unit, compute correlation between its E-targeting weights and the E units' factor selectivity
- Test: Do I units preferentially inhibit E units with specific factor tuning?

### 4. Recurrence Ablation
- After training, set `W_rec = 0` and re-run inference
- Measure drop in emergent selectivity
- If recurrence matters, ablation should eliminate emergent (but not direct) selectivity

### 5. Cross-Seed Consistency
- For E2 and E4, compare I→E weight structure across seeds
- If structure is consistent despite different input assignments, it reflects robust computation

## Implementation Notes

### RF Inference (new function needed)
```python
def infer_rf_locations(neural_data, neuron_types):
    """
    Infer RF location for each neuron based on spatial response profile.
    
    Args:
        neural_data: Trial-averaged responses per location
        neuron_types: Array indicating E (classic) vs I (interneuron)
    
    Returns:
        rf_locations: Array of location indices (or None for I units)
    """
    # For E units: argmax of visual response across locations
    # For I units: return None (they don't get RF-specific input)
```

### Input Tensor Construction
Current shape (assumed): `[n_trials, n_timesteps, n_neurons, n_input_channels]`

New input channels:
- `reward_at_rf`: Available t=0 onward for E units, zero for I units
- `prob_at_rf`: Available t=0 onward for E units, zero for I units  
- `target_in_rf`: Step at target onset for E units
- `target_onset`: Step at target onset for all units (global)
- `salience`: Step at target onset, only for assigned units
- `identity`: Step at target onset, only for assigned units

### Sparse Assignment Storage
Save the random assignment matrices for reproducibility:
```python
assignment = {
    'salience': np.array([...]),  # Boolean, shape [n_neurons]
    'identity': np.array([...]),  # Boolean, shape [n_neurons]
    'seed': seed_value
}
```

## Output Structure

```
results/input_restructuring/
├── E0_baseline/
│   ├── model_weights/
│   ├── metrics.json
│   └── figures/
├── E1_temporal_fix/
│   └── ...
├── E2_primary/
│   ├── seed_0/
│   ├── seed_1/
│   └── ...
├── E3a_sparse_025/
│   └── ...
├── E3b_sparse_075/
│   └── ...
├── E4_replication/
│   └── ...
├── analysis/
│   ├── emergent_selectivity_summary.csv
│   ├── ie_weight_structure.csv
│   ├── cross_seed_consistency.csv
│   └── figures/
│       ├── fit_quality_comparison.png
│       ├── emergent_selectivity_by_experiment.png
│       ├── ie_weight_clustering.png
│       └── ablation_results.png
└── REPORT.md
```

## Success Criteria

1. **Fit quality preserved:** E2 achieves ≥80% of E0's validation correlation
2. **Emergent selectivity detected:** Significant factor selectivity in neurons lacking direct input
3. **Recurrence dependence:** Ablation reduces emergent selectivity by ≥50%
4. **Cross-seed consistency:** I→E weight structure correlates r≥0.5 across seeds
5. **I unit involvement:** I→E weights show non-random structure w.r.t. target E selectivity
