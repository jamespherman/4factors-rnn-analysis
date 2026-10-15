# Context model evaluation (results\recovery_v12\local_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +4.36 | +0.04 / +0.38 / +0.48 | +0.81 / +0.56 / +0.80 | -0.01 / +0.23 / +0.13 | +0.69 | 0.28 |
| free_free | 3 | +3.87 | +0.23 / +0.38 / +0.62 | +1.14 / +0.56 / +0.86 | -0.01 / +0.23 / -0.12 | +0.74 | 0.20 |
| none_free | 3 | +0.00 | +0.02 / +0.38 / -0.04 | -0.03 / +0.56 / -0.17 | +0.02 / +0.23 / +0.29 | +0.27 | 0.15 |
| sharedI_free | 3 | +2.42 | -0.04 / +0.38 / +0.06 | +0.72 / +0.56 / +0.60 | +0.02 / +0.23 / +0.42 | +0.77 | 0.14 |
| shared_free | 3 | +2.57 | +0.11 / +0.38 / +0.01 | +0.97 / +0.56 / +0.59 | -0.00 / +0.23 / +0.24 | +0.81 | 0.33 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.22 / +0.37 | +0.28 / +0.56 | -0.23 / -0.09 | +0.39 / +0.81 |
| free_free | +0.17 / +0.37 | +0.31 / +0.56 | +0.09 / -0.09 | +0.46 / +0.81 |
| none_free | -0.28 / +0.37 | +0.46 / +0.56 | -0.21 / -0.09 | +0.33 / +0.81 |
| sharedI_free | +0.25 / +0.37 | +0.06 / +0.56 | -0.36 / -0.09 | +0.52 / +0.81 |
| shared_free | +0.18 / +0.37 | +0.03 / +0.56 | -0.32 / -0.09 | +0.58 / +0.81 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +5.57 | +0.37 / -0.05 | +0.82 / +0.74 | +0.72 | 1 |
| free_free | +4.81 | -0.22 / -0.05 | +0.87 / +0.74 | +0.72 | 1 |
| none_free | +0.00 | -0.82 / -0.05 | +0.72 / +0.74 | +0.79 | 1 |
| sharedI_free | +3.41 | +0.70 / -0.05 | +0.64 / +0.74 | +0.78 | 1 |
| shared_free | +3.01 | +0.72 / -0.05 | +0.52 / +0.74 | +0.81 | 1 |
