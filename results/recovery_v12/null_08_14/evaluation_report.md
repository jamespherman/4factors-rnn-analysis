# Context model evaluation (results\recovery_v12\null_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +0.01 | +0.18 / +0.05 / +0.27 | +0.53 / +0.03 / +0.23 | -0.01 / -0.15 / -0.18 | +0.54 | 0.42 |
| free_free | 3 | -0.53 | +0.26 / +0.05 / +0.38 | +0.88 / +0.03 / +0.25 | +0.01 / -0.15 / -0.10 | +0.58 | 0.36 |
| none_free | 3 | +0.00 | +0.05 / +0.05 / +0.00 | -0.02 / +0.03 / +0.15 | -0.04 / -0.15 / -0.02 | +0.39 | 0.43 |
| sharedI_free | 3 | +1.30 | +0.07 / +0.05 / -0.08 | -0.11 / +0.03 / +0.44 | -0.06 / -0.15 / -0.18 | +0.31 | 0.40 |
| shared_free | 3 | +1.47 | +0.09 / +0.05 / -0.04 | -0.07 / +0.03 / +0.42 | -0.07 / -0.15 / -0.19 | +0.41 | 0.40 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.18 / -0.24 | +0.03 / -0.19 | -0.21 / -0.17 | +0.47 / +0.87 |
| free_free | -0.34 / -0.24 | -0.02 / -0.19 | -0.16 / -0.17 | +0.49 / +0.87 |
| none_free | -0.21 / -0.24 | -0.07 / -0.19 | -0.11 / -0.17 | +0.36 / +0.87 |
| sharedI_free | -0.22 / -0.24 | +0.07 / -0.19 | +0.04 / -0.17 | +0.27 / +0.87 |
| shared_free | -0.07 / -0.24 | +0.13 / -0.19 | +0.01 / -0.17 | +0.33 / +0.87 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.62 | -0.43 / -0.24 | +0.53 / -0.11 | +0.60 | 1 |
| free_free | -0.15 | -0.59 / -0.24 | +0.64 / -0.11 | +0.58 | 1 |
| none_free | +0.00 | -0.20 / -0.24 | -0.12 / -0.11 | +0.50 | 1 |
| sharedI_free | +3.20 | -0.25 / -0.24 | +0.25 / -0.11 | +0.51 | 1 |
| shared_free | +3.01 | +0.04 / -0.24 | +0.37 / -0.11 | +0.57 | 1 |
