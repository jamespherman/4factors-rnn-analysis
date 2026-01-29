# Recorded vs RNN Neuron Selectivity Figure Plan

## Overview

Create a 1×4 panel figure comparing recorded neuronal selectivity to fitted RNN neuron selectivity across 4 task factors. This will show how well the RNN captures the selectivity patterns of individual neurons.

## Figure Layout

**1 row × 4 columns**, each panel showing:
- **X-axis**: Recorded neuron selectivity (ROC AUC)
- **Y-axis**: RNN neuron selectivity (ROC AUC)
- **Points**: One per neuron, colored by type (E vs I)
- **Reference**: Unity line (y = x) for perfect correspondence

## Panels

| Panel | Factor | High Condition | Low Condition |
|-------|--------|----------------|---------------|
| A | Reward | Reward trial | No-reward trial |
| B | Salience | High salience (face) | Low salience (non-face) |
| C | Probability | High probability (expected target) | Low probability (unexpected target) |
| D | Identity | Face vs Non-face |

**Note**: The "Probability" factor reflects whether the target appeared at the expected (high-probability) or unexpected (low-probability) location within a block. This is distinct from physical location, as the experimental design alternates which locations are high/low probability across blocks.

**Note**: For multi-level factors (identity, salience), we'll use high-vs-low binary comparison where possible, or the maximum absolute selectivity across pairwise comparisons.

## Data Sources

### Sessions
- 08_13: 10 E + 4 I neurons
- 08_14: 25 E + 16 I neurons
- 08_15: 41 E + 13 I neurons
- **Total**: 76 E + 33 I = 109 fitted neurons

### Models
- Full E/I models from `results/replication/` and `results/final_model/`
- Only use neurons that were fitted (first n_E recorded neurons for E, all I neurons)

## Methods

### Time Window

Selectivity is computed using **firing rates averaged in the 50-250ms window after target onset**. This window:
- Excludes the initial transient response (0-50ms)
- Captures the sustained visual/cognitive response
- Avoids late-period motor preparation signals
- With 25ms time bins, this corresponds to bins 2-10 (indices 2:10)

### Selectivity Computation (ROC-based)

For each neuron and factor:

1. **Get trial-averaged firing rates** in the 50-250ms window for the neuron
2. **Binary classification**: High-factor vs Low-factor trials
3. **Compute ROC AUC** using firing rate as classifier
4. **Selectivity index**: `2 * (AUC - 0.5)`, range [-1, 1]
   - +1 = perfect preference for high condition
   - -1 = perfect preference for low condition
   - 0 = no selectivity

### Factor Binarization

| Factor | High (1) | Low (0) | Excluded |
|--------|----------|---------|----------|
| Reward | trial_reward == 1 | trial_reward == 0 | None |
| Salience | trial_salience == high | trial_salience == low | Middle levels |
| Probability | trial_probability == 1 (expected) | trial_probability == 0 (unexpected) | None |
| Identity | Face (1) | Non-face (2) | Bullseye (0) |

### RNN Activity Extraction

1. Load trained model checkpoint
2. Run forward pass on all trials
3. Extract hidden state activations for fitted neurons
4. Compute trial-averaged firing rates
5. Apply same ROC analysis as recorded data

## Implementation Steps

1. **Load data and models** for each session
2. **Extract RNN activations** by running forward pass
3. **Compute selectivity** for both recorded and RNN neurons
4. **Pool across sessions** with neuron type labels
5. **Generate figure** with scatter plots and statistics

## Figure Aesthetics

- **E neurons**: Blue filled circles (`#2166AC`)
- **I neurons**: Orange open circles (`#E66101`)
- **Unity line**: Gray dashed
- **Axis labels**: "Recorded selectivity (ROC)" and "RNN selectivity (ROC)"
- **Panel titles**: Factor names
- **Statistics**: Pearson r and p-value in each panel
- **Figure size**: ~10" × 3" for 1×4 layout

## Output Files

- `results/r01_figure/figure_recorded_vs_rnn.png` (300 DPI)
- `results/r01_figure/figure_recorded_vs_rnn.pdf` (vector)
- `results/r01_figure/panel_data/recorded_vs_rnn_data.npz`

## Script

`scripts/compare_recorded_vs_rnn_selectivity.py`

---

## Results

### Correlation Statistics (50-250ms window, conditioned-loss model)

| Factor | Pearson r | p-value | Interpretation |
|--------|-----------|---------|----------------|
| Reward (goal-directed) | 0.174 | 0.208 | Not significant |
| Salience (high vs low) | 0.115 | 0.408 | Not significant |
| Probability (expected vs unexpected) | -0.086 | 0.537 | Not significant |
| Identity (face vs non-face) | 0.230 | 0.095 | Not significant (trend) |

### Key Observations

1. **Positive trends for Reward, Salience, and Identity**: Unlike the original model which showed negative correlations, the conditioned-loss model shows positive (though non-significant) correlations for 3 of 4 factors.
2. **Identity shows strongest trend** (r=0.230, p=0.095), approaching significance.
3. **Probability shows weak negative correlation** (r=-0.086), indicating expected/unexpected encoding may not transfer well to individual neuron level.
4. **Both E and I neurons** contribute to the scatter plots.

### Interpretation

The lack of significant correlations suggests the RNN is not simply copying recorded selectivity patterns at the individual neuron level. Instead, it finds different solutions that produce similar population-level activity. This is consistent with:
- Degenerate solutions in neural networks
- The RNN capturing population dynamics rather than single-neuron properties
- Factor information being encoded in distributed patterns rather than individual tuning curves

The positive trends (compared to negative correlations in the original model) suggest the condition-specific loss has improved single-neuron selectivity matching, even if the correlations don't reach statistical significance with this sample size.

---

*Plan created: 2025-01-26*
*Results updated: 2026-01-26 (conditioned-loss model, 50-250ms window)*
