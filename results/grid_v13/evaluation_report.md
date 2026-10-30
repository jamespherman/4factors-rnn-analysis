# Context model evaluation (results\grid_v13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeE_free | 6 | -0.55 | +0.39 / +0.32 / +0.63 | +0.39 / +0.32 / +0.49 | -0.17 / -0.74 / +0.23 | +0.60 | 0.34 |
| freeI_free | 6 | -3.54 | +0.35 / +0.32 / +0.63 | +0.41 / +0.32 / +0.48 | -0.04 / -0.74 / +0.03 | +0.62 | 0.43 |
| freeI_free_qshift | 6 | -3.08 | +0.03 / +0.32 / +0.26 | +0.04 / +0.32 / +0.27 | -0.03 / -0.74 / +0.13 | +0.60 | 0.43 |
| free_free | 6 | +1.96 | +0.37 / +0.32 / +0.68 | +0.51 / +0.32 / +0.55 | -0.08 / -0.74 / +0.06 | +0.64 | 0.36 |
| free_free_qshift | 6 | -1.02 | +0.05 / +0.32 / +0.28 | +0.04 / +0.32 / +0.29 | -0.01 / -0.74 / +0.12 | +0.63 | 0.41 |
| none_free | 6 | +0.00 | +0.04 / +0.32 / +0.22 | +0.04 / +0.32 / +0.35 | -0.07 / -0.74 / +0.15 | +0.65 | 0.42 |
| sharedI_free | 6 | +3.03 | +0.11 / +0.32 / +0.29 | +0.19 / +0.32 / +0.27 | -0.02 / -0.74 / +0.06 | +0.67 | 0.40 |
| shared_free | 6 | +2.02 | +0.10 / +0.32 / +0.41 | +0.18 / +0.32 / +0.32 | -0.05 / -0.74 / +0.09 | +0.70 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeE_free | +0.39 / +0.53 | +0.24 / +0.38 | -0.02 / +0.16 | +0.58 / +0.97 |
| freeI_free | +0.38 / +0.53 | +0.26 / +0.38 | -0.11 / +0.16 | +0.60 / +0.97 |
| freeI_free_qshift | +0.21 / +0.53 | +0.11 / +0.38 | -0.00 / +0.16 | +0.58 / +0.97 |
| free_free | +0.45 / +0.53 | +0.23 / +0.38 | -0.08 / +0.16 | +0.61 / +0.97 |
| free_free_qshift | +0.21 / +0.53 | +0.12 / +0.38 | +0.02 / +0.16 | +0.61 / +0.97 |
| none_free | +0.15 / +0.53 | +0.16 / +0.38 | +0.02 / +0.16 | +0.63 / +0.97 |
| sharedI_free | +0.32 / +0.53 | +0.16 / +0.38 | -0.05 / +0.16 | +0.65 / +0.97 |
| shared_free | +0.38 / +0.53 | +0.14 / +0.38 | -0.05 / +0.16 | +0.68 / +0.97 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeE_free | -1.68 | +0.48 / +0.53 | +0.24 / +0.38 | +0.97 | 4 5 5 5 5 5 5 |
| freeI_free | -2.76 | +0.51 / +0.53 | +0.27 / +0.38 | +0.97 | 4 5 5 2 5 5 5 |
| freeI_free_qshift | -5.54 | +0.38 / +0.53 | +0.05 / +0.38 | +0.97 | 4 4 5 4 5 1 5 |
| free_free | -2.71 | +0.45 / +0.53 | +0.26 / +0.38 | +0.97 | 4 4 5 5 5 1 5 |
| free_free_qshift | -4.28 | +0.41 / +0.53 | +0.07 / +0.38 | +0.97 | 5 4 5 4 5 1 5 |
| none_free | +0.00 | +0.29 / +0.53 | +0.07 / +0.38 | +0.97 | 5 5 5 4 5 4 5 |
| sharedI_free | -2.41 | +0.44 / +0.53 | +0.23 / +0.38 | +0.97 | 5 4 5 5 5 5 5 |
| shared_free | -1.68 | +0.37 / +0.53 | +0.06 / +0.38 | +0.97 | 4 5 5 4 5 1 5 |
