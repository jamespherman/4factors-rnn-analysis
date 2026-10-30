# Context model evaluation (results\grid_v13s)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeE_free | 6 | +0.22 | +0.13 / +0.32 / +0.35 | +0.38 / +0.32 / +0.39 | -0.04 / -0.74 / +0.09 | +0.57 | 0.37 |
| freeI_free | 6 | -3.59 | +0.13 / +0.32 / +0.45 | +0.37 / +0.32 / +0.37 | -0.03 / -0.74 / -0.03 | +0.55 | 0.40 |
| free_free | 6 | -1.52 | +0.25 / +0.32 / +0.51 | +0.41 / +0.32 / +0.46 | -0.03 / -0.74 / +0.09 | +0.60 | 0.37 |
| free_free_qshift | 6 | -6.64 | +0.02 / +0.32 / -0.00 | +0.02 / +0.32 / +0.21 | +0.04 / -0.74 / +0.06 | +0.40 | 0.40 |
| none_free | 6 | +0.00 | +0.01 / +0.32 / -0.03 | +0.01 / +0.32 / +0.15 | +0.01 / -0.74 / +0.05 | +0.53 | 0.42 |
| sharedI_free | 6 | -3.23 | +0.01 / +0.32 / +0.23 | +0.09 / +0.32 / +0.19 | +0.01 / -0.74 / +0.07 | +0.49 | 0.42 |
| shared_free | 6 | -3.09 | +0.02 / +0.32 / +0.15 | +0.03 / +0.32 / +0.12 | +0.00 / -0.74 / +0.06 | +0.45 | 0.42 |
| shared_free_qshift | 6 | -0.76 | +0.02 / +0.32 / +0.02 | +0.02 / +0.32 / +0.20 | +0.03 / -0.74 / +0.07 | +0.51 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeE_free | +0.08 / +0.20 | +0.24 / +0.15 | -0.02 / +0.06 | +0.54 / +0.94 |
| freeI_free | +0.17 / +0.20 | +0.22 / +0.15 | -0.06 / +0.06 | +0.52 / +0.94 |
| free_free | +0.17 / +0.20 | +0.23 / +0.15 | -0.05 / +0.06 | +0.57 / +0.94 |
| free_free_qshift | -0.02 / +0.20 | +0.18 / +0.15 | +0.05 / +0.06 | +0.38 / +0.94 |
| none_free | -0.09 / +0.20 | +0.01 / +0.15 | +0.06 / +0.06 | +0.50 / +0.94 |
| sharedI_free | +0.07 / +0.20 | +0.16 / +0.15 | -0.00 / +0.06 | +0.47 / +0.94 |
| shared_free | +0.01 / +0.20 | +0.07 / +0.15 | -0.02 / +0.06 | +0.42 / +0.94 |
| shared_free_qshift | +0.02 / +0.20 | +0.17 / +0.15 | -0.04 / +0.06 | +0.47 / +0.94 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeE_free | +nan | +0.19 / +0.11 | +0.45 / +0.19 | +0.90 | 6 1 4 4 5 1 4 |
| freeI_free | +nan | +0.30 / +0.12 | +0.29 / +0.18 | +0.89 | 6 1 4 4 2 1 4 |
| free_free | +nan | +0.26 / +0.12 | +0.31 / +0.18 | +0.90 | 6 1 4 4 2 1 4 |
| free_free_qshift | +nan | -0.04 / +0.12 | +0.39 / +0.18 | +0.87 | 6 1 4 4 2 1 4 |
| none_free | +nan | -0.10 / +0.12 | +0.24 / +0.18 | +0.86 | 6 1 4 4 2 4 4 |
| sharedI_free | +nan | +0.35 / +0.12 | +0.38 / +0.18 | +0.93 | 6 1 4 4 2 4 4 |
| shared_free | +nan | +0.11 / +0.12 | +0.16 / +0.18 | +0.86 | 6 1 4 1 2 4 4 |
| shared_free_qshift | +nan | +0.20 / +0.12 | +0.47 / +0.18 | +0.93 | 6 1 4 1 2 4 4 |
