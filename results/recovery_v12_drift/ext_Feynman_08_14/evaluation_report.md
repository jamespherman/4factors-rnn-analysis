# Context model evaluation (results\recovery_v12_drift\ext_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -1.17 | +0.01 / +0.08 / +0.15 | +0.28 / +0.66 / +0.47 | +0.02 / +0.52 / +0.16 | +0.14 | 0.44 |
| free_free | 6 | -0.64 | -0.26 / +0.08 / +0.10 | +0.89 / +0.66 / +0.86 | +0.02 / +0.52 / +0.12 | +0.31 | 0.26 |
| none_free | 6 | +0.00 | +0.02 / +0.08 / -0.01 | +0.02 / +0.66 / +0.11 | +0.02 / +0.52 / +0.25 | +0.00 | 0.42 |
| sharedI_free | 6 | -1.28 | +0.07 / +0.08 / -0.16 | +0.21 / +0.66 / +0.57 | +0.02 / +0.52 / +0.10 | +0.22 | 0.32 |
| shared_free | 6 | +0.83 | -0.03 / +0.08 / -0.15 | +0.33 / +0.66 / +0.87 | +0.02 / +0.52 / +0.20 | +0.32 | 0.36 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.13 / +0.24 | +0.16 / +0.41 | +0.38 / -0.09 | -0.17 / -0.47 |
| free_free | +0.02 / +0.24 | +0.54 / +0.41 | +0.12 / -0.09 | -0.07 / -0.47 |
| none_free | +0.04 / +0.24 | -0.21 / +0.41 | +0.00 / -0.09 | -0.02 / -0.47 |
| sharedI_free | +0.22 / +0.24 | +0.32 / +0.41 | +0.42 / -0.09 | -0.21 / -0.47 |
| shared_free | +0.03 / +0.24 | +0.61 / +0.41 | +0.08 / -0.09 | +0.02 / -0.47 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -3.29 | +0.27 / +0.16 | -0.46 / +0.38 | -0.25 | 1 |
| free_free | -1.01 | -0.68 / +0.16 | +0.49 / +0.38 | +0.55 | 4 |
| none_free | +0.00 | +0.64 / +0.16 | -0.10 / +0.38 | -0.42 | 1 |
| sharedI_free | -5.63 | +0.65 / +0.16 | +0.59 / +0.38 | +0.29 | 1 |
| shared_free | -0.71 | -0.67 / +0.16 | +0.61 / +0.38 | +0.62 | 4 |
