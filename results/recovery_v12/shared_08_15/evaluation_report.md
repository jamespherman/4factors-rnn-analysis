# Context model evaluation (results\recovery_v12\shared_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +47.16 | +0.64 / +0.65 / +0.80 | +0.26 / -0.25 / +0.68 | -0.02 / -0.18 / +0.35 | +0.72 | 0.42 |
| free_free | 3 | +40.46 | +0.74 / +0.65 / +0.74 | +0.02 / -0.25 / +0.54 | -0.09 / -0.18 / +0.35 | +0.48 | 0.41 |
| none_free | 3 | +0.00 | +0.00 / +0.65 / +0.12 | +0.03 / -0.25 / -0.02 | +0.02 / -0.18 / +0.23 | +0.48 | 0.41 |
| sharedI_free | 3 | +38.97 | +0.57 / +0.65 / +0.70 | +0.61 / -0.25 / +0.57 | +0.05 / -0.18 / +0.24 | +0.76 | 0.43 |
| shared_free | 3 | +45.68 | +0.44 / +0.65 / +0.79 | +0.20 / -0.25 / +0.44 | -0.03 / -0.18 / +0.25 | +0.67 | 0.47 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.44 / +0.57 | +0.17 / -0.05 | +0.08 / -0.01 | +0.67 / +0.98 |
| free_free | +0.38 / +0.57 | +0.23 / -0.05 | +0.06 / -0.01 | +0.45 / +0.98 |
| none_free | -0.07 / +0.57 | +0.15 / -0.05 | +0.01 / -0.01 | +0.48 / +0.98 |
| sharedI_free | +0.40 / +0.57 | +0.32 / -0.05 | +0.03 / -0.01 | +0.73 / +0.98 |
| shared_free | +0.47 / +0.57 | +0.40 / -0.05 | +0.02 / -0.01 | +0.65 / +0.98 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +85.58 | +0.68 / +0.66 | +0.86 / +0.61 | +1.00 | 1 |
| free_free | +84.03 | +0.53 / +0.66 | +0.88 / +0.61 | +1.00 | 1 |
| none_free | +0.00 | -0.45 / +0.66 | +0.40 / +0.61 | -0.05 | 1 |
| sharedI_free | +84.60 | +0.71 / +0.66 | +0.82 / +0.61 | +1.00 | 1 |
| shared_free | +88.00 | +0.71 / +0.66 | +0.81 / +0.61 | +1.00 | 1 |
