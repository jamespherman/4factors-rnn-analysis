# Context model evaluation (results\verified_grid)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeE_free | 6 | +1.76 | +0.24 / +0.37 / +0.64 | +0.91 / +1.06 / +0.66 | +0.10 / -1.30 / -0.03 | +0.81 | 0.27 |
| freeI_free | 6 | -0.19 | +0.57 / +0.37 / +0.61 | +0.61 / +1.06 / +0.58 | +0.17 / -1.30 / -0.11 | +0.80 | 0.36 |
| free_free | 6 | -9.30 | +0.41 / +0.37 / +0.72 | +1.06 / +1.06 / +0.64 | +0.17 / -1.30 / -0.18 | +0.64 | 0.27 |
| none_free | 6 | +0.00 | +0.02 / +0.37 / -0.05 | +0.17 / +1.06 / +0.21 | +0.02 / -1.30 / +0.06 | +0.84 | 0.31 |
| none_global | 6 | -582.66 | -1.18 / +0.37 / -0.11 | +0.18 / +1.06 / +0.26 | +1.91 / -1.30 / +0.05 | +0.71 | 0.30 |
| none_spatial | 6 | -0.59 | -0.01 / +0.37 / -0.07 | +0.15 / +1.06 / +0.18 | +0.02 / -1.30 / +0.03 | +0.80 | 0.29 |
| sharedE_free | 6 | +1.03 | +0.30 / +0.37 / -0.01 | +0.62 / +1.06 / +0.23 | +0.14 / -1.30 / -0.05 | +0.84 | 0.36 |
| sharedI_free | 6 | +1.81 | +0.33 / +0.37 / +0.08 | +0.51 / +1.06 / +0.28 | +0.17 / -1.30 / -0.08 | +0.86 | 0.37 |
| shared_free | 6 | +0.11 | +0.34 / +0.37 / +0.10 | +0.57 / +1.06 / +0.24 | +0.08 / -1.30 / +0.01 | +0.83 | 0.37 |
| shared_global | 6 | +0.25 | +0.36 / +0.37 / +0.05 | +0.49 / +1.06 / +0.20 | +0.05 / -1.30 / -0.08 | +0.80 | 0.37 |
| shared_spatial | 6 | +1.01 | +0.37 / +0.37 / -0.01 | +0.47 / +1.06 / +0.32 | +0.02 / -1.30 / +0.05 | +0.81 | 0.39 |
| trialvalue_free | 6 | -4.52 | +0.02 / +0.37 / +0.06 | +0.21 / +1.06 / +0.30 | +0.21 / -1.30 / +0.17 | +0.76 | 0.33 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeE_free | +0.36 / +0.64 | +0.24 / +0.46 | -0.12 / +0.12 | +0.79 / +0.97 |
| freeI_free | +0.36 / +0.64 | +0.17 / +0.46 | -0.18 / +0.12 | +0.78 / +0.97 |
| free_free | +0.43 / +0.64 | +0.24 / +0.46 | -0.26 / +0.12 | +0.62 / +0.97 |
| none_free | -0.07 / +0.64 | +0.18 / +0.46 | +0.18 / +0.12 | +0.82 / +0.97 |
| none_global | -0.07 / +0.64 | +0.16 / +0.46 | +0.15 / +0.12 | +0.70 / +0.97 |
| none_spatial | -0.01 / +0.64 | +0.20 / +0.46 | +0.10 / +0.12 | +0.78 / +0.97 |
| sharedE_free | -0.04 / +0.64 | +0.24 / +0.46 | -0.01 / +0.12 | +0.82 / +0.97 |
| sharedI_free | +0.03 / +0.64 | +0.28 / +0.46 | -0.03 / +0.12 | +0.84 / +0.97 |
| shared_free | +0.04 / +0.64 | +0.19 / +0.46 | +0.11 / +0.12 | +0.81 / +0.97 |
| shared_global | -0.01 / +0.64 | +0.26 / +0.46 | -0.02 / +0.12 | +0.79 / +0.97 |
| shared_spatial | -0.03 / +0.64 | +0.18 / +0.46 | +0.08 / +0.12 | +0.80 / +0.97 |
| trialvalue_free | +0.00 / +0.64 | +0.17 / +0.46 | +0.08 / +0.12 | +0.75 / +0.97 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeE_free | -2.05 | +0.51 / +0.64 | +0.34 / +0.46 | +0.97 | 4 4 6 |
| freeI_free | -3.73 | +0.44 / +0.64 | +0.28 / +0.46 | +0.97 | 4 4 6 |
| free_free | -2.49 | +0.48 / +0.64 | +0.20 / +0.46 | +0.96 | 4 5 6 |
| none_free | +0.00 | -0.05 / +0.64 | +0.19 / +0.46 | +0.97 | 6 4 6 |
| none_global | -5.62 | -0.17 / +0.64 | +0.20 / +0.46 | +0.97 | 6 4 6 |
| none_spatial | -1.59 | -0.17 / +0.64 | +0.38 / +0.46 | +0.97 | 5 4 5 |
| sharedE_free | +1.33 | -0.09 / +0.64 | +0.16 / +0.46 | +0.92 | 6 1 5 |
| sharedI_free | -2.39 | -0.05 / +0.64 | +0.08 / +0.46 | +0.95 | 4 5 6 |
| shared_free | -1.84 | -0.05 / +0.64 | -0.06 / +0.46 | +0.96 | 6 5 6 |
| shared_global | -1.83 | -0.09 / +0.64 | +0.04 / +0.46 | +0.96 | 6 5 6 |
| shared_spatial | -2.51 | +0.16 / +0.64 | +0.26 / +0.46 | +0.98 | 5 4 6 |
| trialvalue_free | -3.42 | -0.03 / +0.64 | +0.15 / +0.46 | +0.97 | 4 4 6 |
