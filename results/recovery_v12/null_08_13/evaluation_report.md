# Context model evaluation (results\recovery_v12\null_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +1.22 | +0.09 / +0.21 / +0.38 | +0.64 / +0.13 / -0.26 | +0.13 / +0.19 / +0.14 | +0.43 | 0.41 |
| free_free | 3 | +0.41 | +0.10 / +0.21 / +0.37 | +0.51 / +0.13 / -0.19 | +0.33 / +0.19 / +0.22 | +0.68 | 0.34 |
| none_free | 3 | +0.00 | +0.02 / +0.21 / -0.26 | +0.08 / +0.13 / +0.12 | +0.23 / +0.19 / +0.14 | +0.53 | 0.41 |
| sharedI_free | 3 | -0.24 | +0.07 / +0.21 / -0.04 | +0.53 / +0.13 / -0.28 | +0.21 / +0.19 / +0.15 | +0.57 | 0.38 |
| shared_free | 3 | +0.67 | +0.13 / +0.21 / +0.02 | +0.57 / +0.13 / -0.19 | +0.15 / +0.19 / +0.19 | +0.56 | 0.36 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.01 / +0.05 | +0.27 / -0.45 | +0.21 / +0.01 | +0.41 / +0.97 |
| free_free | +0.11 / +0.05 | +0.22 / -0.45 | -0.06 / +0.01 | +0.67 / +0.97 |
| none_free | -0.21 / +0.05 | +0.12 / -0.45 | +0.15 / +0.01 | +0.50 / +0.97 |
| sharedI_free | -0.14 / +0.05 | +0.26 / -0.45 | +0.23 / +0.01 | +0.53 / +0.97 |
| shared_free | -0.29 / +0.05 | +0.14 / -0.45 | +0.13 / +0.01 | +0.53 / +0.97 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.65 | -0.10 / +0.13 | +0.42 / -0.45 | +0.39 | 1 |
| free_free | +0.71 | -0.26 / +0.13 | +0.44 / -0.45 | +0.55 | 1 |
| none_free | +0.00 | -0.06 / +0.13 | +0.40 / -0.45 | +0.42 | 1 |
| sharedI_free | -0.55 | -0.11 / +0.13 | +0.38 / -0.45 | +0.55 | 1 |
| shared_free | +1.36 | -0.08 / +0.13 | +0.39 / -0.45 | +0.45 | 1 |
