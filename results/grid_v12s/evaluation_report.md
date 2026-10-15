# Context model evaluation (results\grid_v12s)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeE_free | 6 | -3.70 | +0.25 / +0.26 / +0.38 | +0.42 / +0.35 / +0.34 | +0.04 / -0.80 / +0.04 | +0.53 | 0.33 |
| freeI_free | 6 | -4.51 | +0.21 / +0.26 / +0.40 | +0.42 / +0.35 / +0.32 | -0.01 / -0.80 / +0.07 | +0.53 | 0.38 |
| free_free | 6 | -7.66 | +0.21 / +0.26 / +0.35 | +0.50 / +0.35 / +0.36 | +0.01 / -0.80 / +0.04 | +0.47 | 0.33 |
| free_free_qshift | 6 | -14.67 | +0.01 / +0.26 / +0.07 | +0.04 / +0.35 / +0.21 | +0.00 / -0.80 / +0.09 | +0.46 | 0.36 |
| none_free | 6 | +0.00 | +0.01 / +0.26 / +0.03 | +0.03 / +0.35 / +0.22 | +0.01 / -0.80 / +0.06 | +0.59 | 0.39 |
| sharedI_free | 6 | -3.34 | +0.10 / +0.26 / +0.28 | +0.10 / +0.35 / +0.17 | +0.00 / -0.80 / +0.08 | +0.55 | 0.39 |
| shared_free | 6 | -3.04 | +0.14 / +0.26 / +0.25 | +0.13 / +0.35 / +0.16 | +0.02 / -0.80 / +0.05 | +0.54 | 0.39 |
| shared_free_qshift | 6 | -2.63 | +0.02 / +0.26 / +0.09 | +0.04 / +0.35 / +0.22 | +0.02 / -0.80 / +0.09 | +0.63 | 0.40 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeE_free | +0.11 / +0.21 | +0.17 / +0.14 | +0.01 / +0.05 | +0.50 / +0.94 |
| freeI_free | +0.10 / +0.21 | +0.20 / +0.14 | -0.03 / +0.05 | +0.51 / +0.94 |
| free_free | +0.10 / +0.21 | +0.15 / +0.14 | +0.02 / +0.05 | +0.45 / +0.94 |
| free_free_qshift | +0.01 / +0.21 | +0.09 / +0.14 | -0.02 / +0.05 | +0.44 / +0.94 |
| none_free | -0.08 / +0.21 | +0.09 / +0.14 | -0.02 / +0.05 | +0.56 / +0.94 |
| sharedI_free | +0.08 / +0.21 | +0.16 / +0.14 | -0.06 / +0.05 | +0.52 / +0.94 |
| shared_free | +0.10 / +0.21 | +0.18 / +0.14 | -0.04 / +0.05 | +0.52 / +0.94 |
| shared_free_qshift | -0.01 / +0.21 | +0.07 / +0.14 | +0.01 / +0.05 | +0.61 / +0.94 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeE_free | +nan | +0.05 / +0.12 | +0.12 / +0.13 | +0.89 | 6 2 5 4 2 3 4 |
| freeI_free | +nan | +0.17 / +0.28 | +0.36 / +0.08 | +0.89 | 6 2 1 4 2 3 4 |
| free_free | +nan | +0.10 / +0.12 | +0.17 / +0.13 | +0.89 | 6 2 5 4 2 3 4 |
| free_free_qshift | +nan | +0.20 / +0.28 | +0.23 / +0.08 | +0.88 | 6 2 1 4 2 3 4 |
| none_free | +0.00 | -0.08 / +0.22 | +0.53 / +0.10 | +0.93 | 5 2 1 4 2 3 4 |
| sharedI_free | -1.45 | +0.09 / +0.22 | +0.40 / +0.10 | +0.93 | 5 2 1 4 2 3 4 |
| shared_free | +1.06 | +0.12 / +0.22 | +0.33 / +0.10 | +0.93 | 5 2 1 4 2 3 4 |
| shared_free_qshift | +nan | +0.10 / +0.28 | +0.41 / +0.08 | +0.88 | 6 2 1 4 2 3 4 |
