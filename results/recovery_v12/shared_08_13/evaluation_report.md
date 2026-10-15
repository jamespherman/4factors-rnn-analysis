# Context model evaluation (results\recovery_v12\shared_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +2.16 | +0.29 / +0.50 / +0.87 | +0.56 / -0.13 / -0.45 | +0.23 / +0.18 / +0.22 | +0.59 | 0.38 |
| free_free | 3 | +2.07 | +0.34 / +0.50 / +0.93 | +0.38 / -0.13 / -0.37 | +0.36 / +0.18 / +0.21 | +0.76 | 0.38 |
| none_free | 3 | +0.00 | +0.02 / +0.50 / +0.07 | +0.08 / -0.13 / -0.13 | +0.17 / +0.18 / +0.29 | +0.56 | 0.38 |
| sharedI_free | 3 | +1.55 | +0.25 / +0.50 / +0.89 | +0.39 / -0.13 / -0.44 | +0.11 / +0.18 / +0.32 | +0.61 | 0.38 |
| shared_free | 3 | +1.84 | +0.28 / +0.50 / +0.90 | +0.50 / -0.13 / -0.44 | +0.17 / +0.18 / +0.29 | +0.61 | 0.40 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.77 / +0.75 | -0.44 / +0.02 | +0.08 / +0.33 | +0.58 / +0.98 |
| free_free | +0.77 / +0.75 | -0.43 / +0.02 | -0.04 / +0.33 | +0.74 / +0.98 |
| none_free | -0.03 / +0.75 | +0.08 / +0.02 | -0.04 / +0.33 | +0.55 / +0.98 |
| sharedI_free | +0.71 / +0.75 | -0.60 / +0.02 | +0.15 / +0.33 | +0.59 / +0.98 |
| shared_free | +0.68 / +0.75 | -0.56 / +0.02 | +0.20 / +0.33 | +0.58 / +0.98 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | +0.68 / +0.75 | -0.32 / +0.03 | +0.46 | 1 |
| free_free | +nan | +0.73 / +0.75 | -0.22 / +0.03 | +0.68 | 1 |
| none_free | +nan | +0.49 / +0.64 | +0.42 / +0.01 | +1.00 | 3 |
| sharedI_free | +nan | +0.75 / +0.75 | -0.52 / +0.03 | +0.55 | 1 |
| shared_free | +nan | +0.78 / +0.75 | -0.54 / +0.03 | +0.62 | 1 |
