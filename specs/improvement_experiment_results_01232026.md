# Phase 2 Experiment Results
## January 23, 2026

This document logs the results of 17 experiments testing improvements to the E-I RNN model for SC neural data fitting.

---

## Summary

- **Best Result:** `learnable_alpha_neuron` with **0.3449** validation PSTH correlation
- **Improvement over baseline:** +18.9%
- **Target:** 0.3614 (reported best from prior work)
- **Gap remaining:** 4.6%

---

## All Results (Sorted by Performance)

| Rank | Experiment | Best Val Corr | vs Baseline | Epochs | Key Parameters |
|------|------------|---------------|-------------|--------|----------------|
| 1 | **learnable_alpha_neuron** | **0.3449** | **+18.9%** | 451 | E: 0.54±0.07, I: 0.72±0.03 |
| 2 | learnable_alpha_population | 0.3443 | +18.7% | 500 | E: 0.58, I: 0.73 |
| 3 | poisson_ltrial | 0.3369 | +16.1% | 500 | Poisson Bregman distance |
| 4 | low_rank_50 | 0.3334 | +14.9% | 500 | rank=50 |
| 5 | low_rank_20 | 0.3227 | +11.2% | 379 | rank=20 |
| 6 | activity_reg_0.001 | 0.3171 | +9.3% | 500 | λ_activity=0.001 |
| 7 | poisson_ltrial_alpha_pop | 0.3171 | +9.3% | 438 | Combination |
| 8 | low_rank_10 | 0.3167 | +9.2% | 500 | rank=10 |
| 9 | low_rank_20_alpha_pop | 0.3145 | +8.4% | 386 | Combination |
| 10 | low_rank_30 | 0.3120 | +7.6% | 421 | rank=30 |
| 11 | learnable_alpha_scalar | 0.3105 | +7.0% | 500 | α=0.56 |
| 12 | cosine_warmup_50 | 0.3051 | +5.2% | 500 | 50 epoch warmup |
| 13 | **baseline** | **0.2901** | **-** | 500 | Reference |
| 14 | activity_reg_0.01 | 0.2828 | -2.5% | 500 | λ_activity=0.01 (too strong) |
| 15 | cosine_no_warmup | 0.2801 | -3.4% | 500 | No warmup |
| 16 | learnable_tau_neuron_fixed | 0.0234 | -92% | 159 | Broken (α>1) |
| 17 | learnable_tau_population_fixed | 0.0205 | -93% | 163 | Broken (α>1) |

---

## Detailed Analysis

### 1. Learnable Alpha (Best Improvement)

**Winner: Per-neuron learnable alpha (+18.9%)**

The learnable alpha feature allows the integration constant α = dt/τ to be learned directly, using sigmoid parameterization to keep α ∈ (0, 1).

**Learned values (per-neuron):**
- Excitatory neurons: α_e = 0.54 ± 0.07 (slower than default 0.5)
- Inhibitory neurons: α_i = 0.72 ± 0.03 (faster than default)

**Key insight:** The optimal dynamics require E neurons to integrate more slowly and I neurons to integrate faster than the default τ=50ms assumption.

### 2. Poisson L_trial (+16.1%)

Using Poisson Bregman divergence instead of Euclidean distance for trial matching:

```
D(target || model) = target * log(target/model) - target + model
```

This respects the natural variance structure of spike count data where variance scales with mean.

### 3. Low-Rank Connectivity (+14.9% at rank=50)

Constraining W_rec = U @ V^T provides regularization and reduces parameters.

**Non-monotonic relationship with rank:**
- rank=10: 0.3167
- rank=20: 0.3227
- rank=30: 0.3120
- rank=50: 0.3334 (best)

### 4. Learnable Tau (FAILED)

Both population and per-neuron learnable tau implementations failed catastrophically despite using soft clamping (softplus).

**Root cause:** When tau_i ≈ 20ms < dt = 25ms, the resulting α_i = dt/τ > 1, which breaks the RNN dynamics. The state update becomes:
```
x = (1 - α) * x + α * input
```
When α > 1, this is no longer a valid exponential moving average.

**Fix needed:** Constrain tau_min ≥ dt (25ms) instead of tau_min = 10ms.

### 5. Scheduler Effects

- **Cosine without warmup:** Hurt performance (-3.4%)
- **Cosine with 50-epoch warmup:** Small improvement (+5.2%)
- **ReduceLROnPlateau (baseline):** Best scheduler

### 6. Activity Regularization

- **λ=0.01:** Too strong, hurt performance (-2.5%)
- **λ=0.001:** Marginal benefit (+9.3%)

### 7. Combinations Tested

Only 2 combinations were tested:

| Combination | Result | vs Best Individual |
|-------------|--------|-------------------|
| poisson_ltrial + alpha_pop | 0.3171 | < 0.3443 (alpha_pop) |
| low_rank_20 + alpha_pop | 0.3145 | < 0.3443 (alpha_pop) |

**Observation:** These specific combinations did not stack benefits. However, many other combinations were NOT tested.

---

## Code Changes Made

### src/model.py
1. **Learnable alpha** with logit parameterization (sigmoid bounds)
2. **Soft clamping for tau** using softplus (still broken due to tau < dt)
3. **Low-rank connectivity** via W_rec = U @ V^T factorization
4. New methods: `get_alpha_values()`, `_alpha_to_logit()`, `_logit_to_alpha()`

### src/losses.py
1. **Poisson Bregman divergence** option in `compute_L_trial()`
2. Parameter: `use_poisson_distance=True`

### scripts/experiment_phase2.py
Complete experiment script with 17 configurations.

---

## Files Generated

```
results/phase2/
├── summary.json                           # All results
├── comparison.png                         # Visualization
├── baseline_result.json
├── poisson_ltrial_result.json
├── learnable_alpha_scalar_result.json
├── learnable_alpha_population_result.json
├── learnable_alpha_neuron_result.json
├── learnable_tau_population_fixed_result.json
├── learnable_tau_neuron_fixed_result.json
├── low_rank_10_result.json
├── low_rank_20_result.json
├── low_rank_30_result.json
├── low_rank_50_result.json
├── cosine_no_warmup_result.json
├── cosine_warmup_50_result.json
├── activity_reg_0.01_result.json
├── activity_reg_0.001_result.json
├── poisson_ltrial_alpha_pop_result.json
└── low_rank_20_alpha_pop_result.json
```

---

## Next Steps

See `specs/improvement_plan_01232026c.md` for the detailed plan to achieve further improvement.

**Priority combinations to test:**
1. learnable_alpha_neuron + poisson_ltrial (both top performers, not yet combined)
2. learnable_alpha_neuron + low_rank_50
3. Triple combination: alpha_neuron + poisson_ltrial + low_rank

**New ideas to explore:**
1. Input embedding expansion (Cover's theorem)
2. Fix learnable_tau by constraining tau_min ≥ dt
3. Initialize alpha closer to learned optimal values (E: 0.54, I: 0.72)
