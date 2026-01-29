# Phase 2c Experiment Results
## January 23, 2026

This document logs the results of 16 experiments testing systematic combinations and input embedding for the E-I RNN model.

---

## Executive Summary

**Major Breakthrough: Input embedding (Cover's theorem) achieved 0.3762 validation PSTH correlation - exceeding the target of 0.3614 by 4.1%.**

| Milestone | Val Correlation | Change |
|-----------|-----------------|--------|
| Phase 1 baseline | 0.2901 | - |
| Phase 2 best | 0.3449 | +18.9% |
| Target (prior work) | 0.3614 | +24.6% |
| **Phase 2c best** | **0.3762** | **+29.7%** |

---

## All Results (Sorted by Performance)

| Rank | Experiment | Best Val Corr | vs Phase 2 | Key Finding |
|------|------------|---------------|------------|-------------|
| 1 | **embed_56_learnable** | **0.3762** | **+9.1%** | 4x input expansion |
| 2 | alpha_init_optimal | 0.3536 | +2.5% | Better initialization |
| 3 | noise_0.15 | 0.3501 | +1.5% | Higher noise helps |
| 4 | learnable_tau_pop_v2 | 0.3461 | +0.3% | Fixed tau bug |
| 5 | noise_0.05 | 0.3377 | -2.1% | Lower noise hurts |
| 6 | embed_time_lag | 0.3365 | -2.4% | Time lag features |
| 7 | embed_28_learnable | 0.3341 | -3.1% | 2x input expansion |
| 8 | embed_28_random | 0.3245 | -5.9% | Random projection |
| 9 | embed_28_alpha_neuron | 0.3167 | -8.2% | Combination hurt |
| 10 | alpha_neuron_poisson_actreg | 0.3105 | -10.0% | Triple combo hurt |
| 11 | alpha_neuron_poisson | 0.3083 | -10.6% | Combination hurt |
| 12 | alpha_neuron_poisson_optimal_init | 0.3007 | -12.8% | Combination hurt |
| 13 | poisson_lowrank50 | 0.2929 | -15.1% | Combination hurt |
| 14 | fixed_alpha_optimal | 0.2901 | -15.9% | Same as baseline |
| 15 | alpha_neuron_lowrank50 | 0.2782 | -19.3% | Combination hurt |
| 16 | alpha_neuron_poisson_lowrank50 | 0.2746 | -20.4% | Triple combo worst |

---

## Key Findings

### 1. Input Embedding is the Clear Winner

**embed_56_learnable** achieved **0.3762** - a breakthrough result that:
- Exceeds the target (0.3614) by 4.1%
- Improves over Phase 2 best by 9.1%
- Uses a simple learnable linear expansion: 14 inputs -> 56 dimensions + ReLU

**Why it works (Cover's Theorem):** Expanding inputs to a higher-dimensional space via a nonlinear transformation makes complex patterns more linearly separable. The RNN can more easily learn to represent task variables when they're projected into a richer feature space.

**Embedding comparison:**
| Embedding Type | Dimension | Val Corr | Notes |
|----------------|-----------|----------|-------|
| Learnable (4x) | 14 -> 56 | 0.3762 | Best |
| Time-lagged | 14 -> 56 | 0.3365 | Fixed features |
| Learnable (2x) | 14 -> 28 | 0.3341 | Good but less |
| Random | 14 -> 28 | 0.3245 | Fixed projection |

### 2. Combinations Consistently Hurt Performance

**Every combination tested performed worse than either component alone:**

| Combination | Result | Best Component |
|-------------|--------|----------------|
| alpha_neuron + poisson_ltrial | 0.3083 | 0.3449 (alpha_neuron) |
| alpha_neuron + low_rank_50 | 0.2782 | 0.3449 (alpha_neuron) |
| poisson_ltrial + low_rank_50 | 0.2929 | 0.3369 (poisson_ltrial) |
| All three combined | 0.2746 | Worst of all |

**Why combinations fail:**
- The improvements address similar underlying limitations (expressiveness vs regularization)
- Adding multiple changes simultaneously may create conflicting optimization pressures
- The gradient balancing scheme may not work well with multiple loss modifications
- Each improvement may have been tuned for the baseline, not for combination

### 3. Fixed Learnable Tau Now Works

**learnable_tau_pop_v2: 0.3461** - After fixing TAU_MIN >= dt (25ms), learnable tau performs comparably to learnable_alpha_population (0.3443).

Learned values:
- tau_e: 43.3ms (alpha_e = 0.578)
- tau_i: 35.1ms (alpha_i = 0.713)

These match the alpha values learned in Phase 2, confirming the fix was successful.

### 4. Optimal Alpha Initialization Helps

**alpha_init_optimal: 0.3536** - Starting alpha closer to learned optimal values (E: 0.54, I: 0.72) improves performance over the default initialization, though not as much as input embedding.

### 5. Noise Scale Effects

| Noise Scale | Val Corr | vs Default (0.1) |
|-------------|----------|------------------|
| 0.05 | 0.3377 | -2.1% |
| 0.10 (default) | 0.3449 | baseline |
| 0.15 | 0.3501 | +1.5% |

Slightly higher noise (0.15) marginally improves performance, possibly by preventing overfitting or improving trial-to-trial variability matching.

---

## Implementation Details

### Input Embedding (src/model.py)

```python
class InputEmbedding(nn.Module):
    def __init__(self, n_inputs, embed_dim, embed_type='learnable'):
        if embed_type == 'learnable':
            self.linear = nn.Linear(n_inputs, embed_dim)

    def forward(self, x):
        return F.relu(self.linear(x))
```

The embedding is applied at the start of the forward pass:
```python
if self.input_embed is not None:
    inputs = self.input_embed(inputs)
```

### Fixed TAU_MIN

Changed from 10ms to 25ms to ensure alpha = dt/tau < 1:
```python
TAU_MIN = 25.0  # Must be >= dt to prevent alpha > 1
```

---

## Best Configuration

```python
{
    'name': 'embed_56_learnable',
    'input_embed_dim': 56,
    'input_embed_type': 'learnable',
    'loss_type': 'correlation',
    'scheduler': 'plateau',
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'max_epochs': 500,
    'patience': 100
}
```

This configuration:
- Uses default alpha/tau (no learnable dynamics)
- Uses default noise_scale (0.1)
- Uses gradient balancing with L_trial scale = 0.5
- Uses ReduceLROnPlateau scheduler

---

## Files Generated

```
results/phase2c/
├── summary.json
├── comparison.png
├── embed_56_learnable_result.json   <- BEST
├── alpha_init_optimal_result.json
├── noise_0.15_result.json
├── learnable_tau_pop_v2_result.json
├── noise_0.05_result.json
├── embed_time_lag_result.json
├── embed_28_learnable_result.json
├── embed_28_random_result.json
├── embed_28_alpha_neuron_result.json
├── alpha_neuron_poisson_actreg_result.json
├── alpha_neuron_poisson_result.json
├── alpha_neuron_poisson_optimal_init_result.json
├── poisson_lowrank50_result.json
├── fixed_alpha_optimal_result.json
├── alpha_neuron_lowrank50_result.json
└── alpha_neuron_poisson_lowrank50_result.json
```

---

## Summary Statistics

| Category | Experiments | Above Target | Above Phase 2 Best |
|----------|-------------|--------------|-------------------|
| Input Embedding | 5 | 1 (20%) | 1 (20%) |
| Combinations | 6 | 0 (0%) | 0 (0%) |
| Hyperparameters | 5 | 0 (0%) | 3 (60%) |
| **Total** | **16** | **1 (6.3%)** | **4 (25%)** |

---

## Next Steps

See `specs/phase3_improvement_recommendations.md` for detailed recommendations on how to further improve model fits.
