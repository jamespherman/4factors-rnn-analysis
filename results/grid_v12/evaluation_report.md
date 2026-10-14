# Context model evaluation (results\grid_v12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeE_free | 6 | +1.56 | +0.27 / +0.26 / +0.71 | +0.49 / +0.35 / +0.56 | -0.15 / -0.80 / +0.20 | +0.76 | 0.32 |
| freeI_free | 6 | -1.23 | +0.27 / +0.26 / +0.64 | +0.44 / +0.35 / +0.50 | -0.09 / -0.80 / +0.14 | +0.66 | 0.40 |
| freeI_free_qshift | 6 | -2.11 | +0.04 / +0.26 / +0.35 | +0.08 / +0.35 / +0.31 | -0.02 / -0.80 / +0.08 | +0.72 | 0.41 |
| free_free | 6 | +1.37 | +0.34 / +0.26 / +0.71 | +0.49 / +0.35 / +0.55 | -0.18 / -0.80 / +0.25 | +0.76 | 0.33 |
| free_free_qshift | 6 | -2.57 | +0.03 / +0.26 / +0.43 | +0.07 / +0.35 / +0.31 | -0.06 / -0.80 / +0.18 | +0.77 | 0.40 |
| none_free | 6 | +0.00 | +0.03 / +0.26 / +0.19 | +0.05 / +0.35 / +0.26 | -0.01 / -0.80 / +0.11 | +0.68 | 0.40 |
| sharedI_free | 6 | -0.86 | +0.11 / +0.26 / +0.39 | +0.10 / +0.35 / +0.24 | +0.00 / -0.80 / +0.08 | +0.75 | 0.40 |
| shared_free | 6 | +0.84 | +0.11 / +0.26 / +0.39 | +0.07 / +0.35 / +0.23 | -0.02 / -0.80 / +0.13 | +0.78 | 0.37 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeE_free | +0.50 / +0.55 | +0.21 / +0.39 | -0.03 / +0.15 | +0.74 / +0.98 |
| freeI_free | +0.41 / +0.55 | +0.23 / +0.39 | -0.04 / +0.15 | +0.64 / +0.98 |
| freeI_free_qshift | +0.30 / +0.55 | +0.13 / +0.39 | -0.07 / +0.15 | +0.70 / +0.98 |
| free_free | +0.47 / +0.55 | +0.23 / +0.39 | -0.02 / +0.15 | +0.74 / +0.98 |
| free_free_qshift | +0.35 / +0.55 | +0.12 / +0.39 | +0.01 / +0.15 | +0.75 / +0.98 |
| none_free | +0.13 / +0.55 | +0.14 / +0.39 | -0.05 / +0.15 | +0.66 / +0.98 |
| sharedI_free | +0.39 / +0.55 | +0.05 / +0.39 | -0.07 / +0.15 | +0.73 / +0.98 |
| shared_free | +0.36 / +0.55 | +0.06 / +0.39 | -0.00 / +0.15 | +0.76 / +0.98 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeE_free | +1.38 | +0.51 / +0.55 | +0.25 / +0.39 | +0.97 | 4 4 5 5 5 2 6 |
| freeI_free | +1.89 | +0.52 / +0.55 | +0.21 / +0.39 | +0.98 | 4 4 5 4 5 3 5 |
| freeI_free_qshift | +0.13 | +0.43 / +0.55 | +0.08 / +0.39 | +0.97 | 5 4 6 4 5 6 6 |
| free_free | +1.38 | +0.50 / +0.55 | +0.24 / +0.39 | +0.97 | 4 4 5 3 4 2 5 |
| free_free_qshift | -0.52 | +0.39 / +0.55 | +0.01 / +0.39 | +0.97 | 5 4 6 4 5 2 6 |
| none_free | +0.00 | +0.18 / +0.55 | +0.04 / +0.39 | +0.97 | 5 4 6 4 4 6 6 |
| sharedI_free | +0.67 | +0.50 / +0.55 | +0.04 / +0.39 | +0.97 | 5 4 5 4 4 2 5 |
| shared_free | +2.34 | +0.46 / +0.55 | +0.04 / +0.39 | +0.98 | 5 4 5 4 4 6 6 |
