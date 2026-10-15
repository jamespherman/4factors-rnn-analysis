# Context model evaluation (results\recovery_v12\local_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +13.44 | +0.66 / +0.83 / +0.73 | +0.67 / +0.61 / +0.63 | -0.16 / -1.15 / -0.08 | +0.62 | 0.41 |
| free_free | 3 | +5.78 | +0.89 / +0.83 / +0.84 | +0.82 / +0.61 / +0.82 | -0.13 / -1.15 / -0.14 | +0.74 | 0.38 |
| none_free | 3 | +0.00 | -0.08 / +0.83 / +0.12 | +0.19 / +0.61 / +0.46 | +0.00 / -1.15 / +0.10 | +0.85 | 0.39 |
| sharedI_free | 3 | +13.46 | +0.39 / +0.83 / +0.63 | +0.70 / +0.61 / +0.77 | -0.18 / -1.15 / +0.34 | +0.88 | 0.42 |
| shared_free | 3 | +15.00 | +0.25 / +0.83 / +0.53 | +0.58 / +0.61 / +0.46 | -0.08 / -1.15 / -0.02 | +0.80 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.46 / +0.50 | +0.31 / +0.40 | -0.17 / +0.24 | +0.61 / +0.99 |
| free_free | +0.53 / +0.50 | +0.52 / +0.40 | -0.25 / +0.24 | +0.74 / +0.99 |
| none_free | -0.03 / +0.50 | +0.32 / +0.40 | -0.01 / +0.24 | +0.84 / +0.99 |
| sharedI_free | +0.19 / +0.50 | +0.49 / +0.40 | +0.18 / +0.24 | +0.86 / +0.99 |
| shared_free | +0.47 / +0.50 | +0.39 / +0.40 | -0.10 / +0.24 | +0.80 / +0.99 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.78 | +0.73 / +0.67 | +0.85 / +0.78 | +1.00 | 1 |
| free_free | +3.12 | +0.67 / +0.67 | +0.84 / +0.78 | +1.00 | 1 |
| none_free | +0.00 | -0.53 / +0.67 | +0.61 / +0.78 | +0.99 | 1 |
| sharedI_free | +1.76 | +0.75 / +0.67 | +0.81 / +0.78 | +1.00 | 1 |
| shared_free | +0.09 | +0.73 / +0.67 | +0.79 / +0.78 | +1.00 | 1 |
