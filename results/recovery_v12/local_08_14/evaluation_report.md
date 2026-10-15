# Context model evaluation (results\recovery_v12\local_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +0.92 | +0.52 / +0.63 / +0.80 | +0.30 / -0.06 / +0.81 | +0.02 / -0.30 / -0.25 | +0.47 | 0.39 |
| free_free | 3 | +2.21 | +0.68 / +0.63 / +0.82 | +0.43 / -0.06 / +0.75 | +0.05 / -0.30 / -0.29 | +0.45 | 0.31 |
| none_free | 3 | +0.00 | +0.01 / +0.63 / +0.42 | -0.06 / -0.06 / +0.58 | -0.01 / -0.30 / -0.16 | +0.46 | 0.43 |
| sharedI_free | 3 | +0.39 | +0.08 / +0.63 / +0.39 | -0.15 / -0.06 / +0.73 | +0.04 / -0.30 / -0.31 | +0.47 | 0.47 |
| shared_free | 3 | +1.02 | +0.03 / +0.63 / +0.33 | -0.18 / -0.06 / +0.68 | +0.00 / -0.30 / -0.23 | +0.44 | 0.46 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.44 / +0.53 | +0.43 / +0.44 | -0.20 / +0.19 | +0.31 / +0.80 |
| free_free | +0.46 / +0.53 | +0.37 / +0.44 | -0.25 / +0.19 | +0.31 / +0.80 |
| none_free | +0.09 / +0.53 | +0.28 / +0.44 | -0.15 / +0.19 | +0.27 / +0.80 |
| sharedI_free | +0.06 / +0.53 | +0.34 / +0.44 | -0.22 / +0.19 | +0.26 / +0.80 |
| shared_free | +0.02 / +0.53 | +0.29 / +0.44 | -0.20 / +0.19 | +0.26 / +0.80 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | +0.39 / +0.47 | +0.31 / +0.42 | +0.48 | 1 |
| free_free | +nan | +0.52 / +0.47 | +0.30 / +0.42 | +0.59 | 1 |
| none_free | +nan | -0.05 / +0.39 | +0.05 / +0.28 | +0.24 | 3 |
| sharedI_free | +nan | -0.28 / +0.47 | +0.37 / +0.42 | +0.50 | 1 |
| shared_free | +nan | +0.10 / +0.39 | +0.10 / +0.28 | +0.40 | 3 |
