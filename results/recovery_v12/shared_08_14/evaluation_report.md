# Context model evaluation (results\recovery_v12\shared_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -0.21 | +0.65 / +0.62 / +0.92 | +0.31 / -0.15 / +0.45 | -0.05 / +0.04 / +0.19 | +0.69 | 0.47 |
| free_free | 3 | -1.74 | +0.75 / +0.62 / +0.92 | +0.41 / -0.15 / +0.44 | -0.01 / +0.04 / +0.18 | +0.68 | 0.46 |
| none_free | 3 | +0.00 | +0.11 / +0.62 / +0.64 | +0.01 / -0.15 / +0.46 | -0.04 / +0.04 / +0.24 | +0.67 | 0.49 |
| sharedI_free | 3 | -0.45 | +0.53 / +0.62 / +0.93 | -0.07 / -0.15 / +0.75 | -0.01 / +0.04 / +0.25 | +0.65 | 0.48 |
| shared_free | 3 | +0.47 | +0.51 / +0.62 / +0.93 | -0.09 / -0.15 / +0.74 | -0.00 / +0.04 / +0.18 | +0.69 | 0.48 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.74 / +0.77 | +0.21 / -0.03 | -0.00 / -0.24 | +0.47 / +0.81 |
| free_free | +0.73 / +0.77 | +0.21 / -0.03 | -0.07 / -0.24 | +0.44 / +0.81 |
| none_free | +0.48 / +0.77 | +0.17 / -0.03 | +0.01 / -0.24 | +0.45 / +0.81 |
| sharedI_free | +0.77 / +0.77 | +0.33 / -0.03 | +0.06 / -0.24 | +0.42 / +0.81 |
| shared_free | +0.77 / +0.77 | +0.32 / -0.03 | +0.12 / -0.24 | +0.46 / +0.81 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | +0.68 / +0.64 | +0.07 / +0.10 | +0.96 | 3 |
| free_free | +nan | +0.66 / +0.64 | +0.11 / +0.10 | +0.98 | 3 |
| none_free | +0.00 | +0.36 / +0.78 | -0.26 / -0.61 | +0.48 | 1 |
| sharedI_free | +nan | +0.71 / +0.64 | +0.72 / +0.10 | +0.99 | 3 |
| shared_free | -0.17 | +0.75 / +0.78 | -0.62 / -0.61 | +0.54 | 1 |
