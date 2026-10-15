# Context model evaluation (results\recovery_v12\local_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +5.97 | +0.39 / +0.54 / +0.78 | +0.73 / +0.16 / +0.63 | +0.16 / +0.41 / +0.07 | +0.56 | 0.36 |
| free_free | 3 | +6.12 | +0.19 / +0.54 / +0.73 | +0.58 / +0.16 / +0.70 | +0.32 / +0.41 / +0.17 | +0.74 | 0.25 |
| none_free | 3 | +0.00 | +0.03 / +0.54 / +0.25 | +0.02 / +0.16 / -0.22 | +0.16 / +0.41 / +0.05 | +0.62 | 0.41 |
| sharedI_free | 3 | +4.32 | +0.05 / +0.54 / +0.55 | +0.37 / +0.16 / +0.56 | +0.21 / +0.41 / +0.08 | +0.66 | 0.38 |
| shared_free | 3 | +5.45 | +0.04 / +0.54 / +0.57 | +0.29 / +0.16 / +0.57 | +0.38 / +0.41 / +0.16 | +0.74 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.63 / +0.79 | +0.30 / +0.38 | +0.15 / -0.01 | +0.50 / +0.97 |
| free_free | +0.55 / +0.79 | +0.37 / +0.38 | +0.18 / -0.01 | +0.69 / +0.97 |
| none_free | +0.22 / +0.79 | -0.24 / +0.38 | +0.16 / -0.01 | +0.60 / +0.97 |
| sharedI_free | +0.46 / +0.79 | +0.24 / +0.38 | +0.23 / -0.01 | +0.61 / +0.97 |
| shared_free | +0.45 / +0.79 | +0.26 / +0.38 | +0.17 / -0.01 | +0.69 / +0.97 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | +0.70 / +0.76 | +0.29 / +0.44 | +0.54 | 3 |
| free_free | +nan | +0.47 / +0.76 | +0.34 / +0.44 | +0.79 | 3 |
| none_free | +0.00 | +0.25 / +0.83 | -0.25 / +0.31 | +0.79 | 1 |
| sharedI_free | +nan | +0.19 / +0.76 | +0.31 / +0.44 | +0.81 | 3 |
| shared_free | +nan | +0.22 / +0.76 | +0.31 / +0.44 | +0.87 | 3 |
