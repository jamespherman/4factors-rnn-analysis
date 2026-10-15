# Context model evaluation (results\recovery_v12\ext_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +5.76 | +0.24 / +0.54 / +0.70 | +0.58 / +0.23 / +0.51 | +0.21 / +0.43 / +0.27 | +0.65 | 0.37 |
| free_free | 3 | +7.04 | +0.52 / +0.54 / +0.89 | +0.52 / +0.23 / +0.67 | +0.22 / +0.43 / +0.31 | +0.70 | 0.34 |
| none_free | 3 | +0.00 | -0.01 / +0.54 / +0.05 | +0.03 / +0.23 / -0.07 | +0.24 / +0.43 / +0.18 | +0.68 | 0.50 |
| sharedI_free | 3 | +3.89 | +0.06 / +0.54 / +0.52 | +0.40 / +0.23 / +0.52 | +0.31 / +0.43 / +0.29 | +0.69 | 0.40 |
| shared_free | 3 | +5.57 | +0.03 / +0.54 / +0.43 | +0.45 / +0.23 / +0.47 | -0.00 / +0.43 / +0.14 | +0.63 | 0.37 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.59 / +0.63 | +0.42 / +0.54 | +0.33 / +0.12 | +0.61 / +0.96 |
| free_free | +0.65 / +0.63 | +0.56 / +0.54 | +0.32 / +0.12 | +0.69 / +0.96 |
| none_free | +0.22 / +0.63 | +0.06 / +0.54 | +0.22 / +0.12 | +0.64 / +0.96 |
| sharedI_free | +0.42 / +0.63 | +0.48 / +0.54 | +0.40 / +0.12 | +0.67 / +0.96 |
| shared_free | +0.36 / +0.63 | +0.44 / +0.54 | +0.35 / +0.12 | +0.61 / +0.96 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | +0.51 / +0.42 | +0.40 / +0.57 | +0.79 | 3 |
| free_free | +nan | +0.60 / +0.42 | +0.62 / +0.57 | +0.99 | 3 |
| none_free | +nan | -0.02 / +0.42 | -0.24 / +0.57 | +0.81 | 3 |
| sharedI_free | +nan | +0.12 / +0.42 | +0.46 / +0.57 | +0.90 | 3 |
| shared_free | +nan | +0.03 / +0.42 | +0.39 / +0.57 | +0.75 | 3 |
