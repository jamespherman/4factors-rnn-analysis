# Context model evaluation (results\recovery_v12\local_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +1.82 | +0.14 / -0.03 / +0.61 | +0.44 / -0.15 / +0.60 | -0.05 / +0.04 / +0.21 | +0.80 | 0.36 |
| free_free | 3 | +1.71 | +0.18 / -0.03 / +0.68 | +0.38 / -0.15 / +0.78 | -0.05 / +0.04 / +0.15 | +0.89 | 0.29 |
| none_free | 3 | +0.00 | -0.00 / -0.03 / +0.11 | +0.04 / -0.15 / +0.70 | -0.05 / +0.04 / +0.13 | +0.72 | 0.39 |
| sharedI_free | 3 | +0.53 | -0.01 / -0.03 / +0.15 | +0.05 / -0.15 / +0.39 | -0.08 / +0.04 / +0.03 | +0.74 | 0.39 |
| shared_free | 3 | +0.62 | -0.01 / -0.03 / +0.15 | +0.07 / -0.15 / +0.33 | -0.08 / +0.04 / -0.00 | +0.73 | 0.39 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.28 / +0.19 | +0.42 / +0.60 | +0.03 / +0.13 | +0.74 / +0.89 |
| free_free | +0.19 / +0.19 | +0.50 / +0.60 | +0.07 / +0.13 | +0.82 / +0.89 |
| none_free | +0.29 / +0.19 | +0.55 / +0.60 | +0.07 / +0.13 | +0.68 / +0.89 |
| sharedI_free | +0.02 / +0.19 | +0.44 / +0.60 | +0.04 / +0.13 | +0.70 / +0.89 |
| shared_free | +0.03 / +0.19 | +0.40 / +0.60 | +0.05 / +0.13 | +0.69 / +0.89 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.69 | +0.15 / +0.17 | +0.44 / +0.73 | +0.78 | 2 |
| free_free | +1.29 | +0.16 / +0.17 | +0.64 / +0.73 | +0.85 | 2 |
| none_free | +0.00 | +0.34 / +0.17 | +0.57 / +0.73 | +0.78 | 2 |
| sharedI_free | -0.26 | +0.09 / +0.17 | +0.35 / +0.73 | +0.83 | 2 |
| shared_free | -0.50 | +0.17 / +0.17 | +0.30 / +0.73 | +0.84 | 2 |
