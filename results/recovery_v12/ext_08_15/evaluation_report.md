# Context model evaluation (results\recovery_v12\ext_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -52.17 | +0.91 / +0.85 / +0.55 | +0.65 / +0.44 / +0.40 | +0.06 / -1.39 / -0.18 | +0.45 | 0.42 |
| free_free | 3 | -9.49 | +0.87 / +0.85 / +0.92 | +0.44 / +0.44 / +0.89 | -0.46 / -1.39 / +0.25 | +0.89 | 0.36 |
| none_free | 3 | +0.00 | +0.04 / +0.85 / +0.10 | +0.04 / +0.44 / +0.28 | -0.04 / -1.39 / +0.32 | +0.66 | 0.36 |
| sharedI_free | 3 | -1.40 | +0.54 / +0.85 / +0.30 | +0.71 / +0.44 / +0.29 | -0.04 / -1.39 / +0.20 | +0.69 | 0.45 |
| shared_free | 3 | -2.48 | +0.53 / +0.85 / +0.46 | +0.66 / +0.44 / +0.39 | -0.06 / -1.39 / +0.11 | +0.85 | 0.45 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.39 / +0.53 | +0.28 / +0.47 | -0.18 / +0.20 | +0.43 / +0.98 |
| free_free | +0.56 / +0.53 | +0.48 / +0.47 | +0.07 / +0.20 | +0.89 / +0.98 |
| none_free | +0.04 / +0.53 | +0.28 / +0.47 | +0.06 / +0.20 | +0.65 / +0.98 |
| sharedI_free | +0.30 / +0.53 | +0.26 / +0.47 | -0.04 / +0.20 | +0.69 / +0.98 |
| shared_free | +0.50 / +0.53 | +0.37 / +0.47 | -0.11 / +0.20 | +0.84 / +0.98 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -96.48 | +0.44 / +0.70 | +0.62 / +0.78 | +0.72 | 1 |
| free_free | +4.60 | +0.71 / +0.70 | +0.84 / +0.78 | +1.00 | 1 |
| none_free | +0.00 | -0.32 / +0.70 | +0.67 / +0.78 | +0.99 | 1 |
| sharedI_free | +2.25 | +0.25 / +0.70 | +0.64 / +0.78 | +0.99 | 1 |
| shared_free | +0.35 | +0.72 / +0.70 | +0.86 / +0.78 | +0.99 | 1 |
