# Context model evaluation (results\recovery_v12_drift\null_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +2.06 | +0.12 / +0.06 / +0.24 | +1.65 / +0.85 / +0.79 | +0.06 / +0.60 / +0.14 | +0.84 | 0.46 |
| free_free | 6 | +2.79 | +0.15 / +0.06 / +0.30 | +1.87 / +0.85 / +0.82 | +0.12 / +0.60 / +0.21 | +0.85 | 0.37 |
| none_free | 6 | +0.00 | +0.07 / +0.06 / -0.05 | +0.17 / +0.85 / +0.64 | -0.03 / +0.60 / -0.08 | +0.77 | 0.42 |
| sharedI_free | 6 | +2.73 | +0.00 / +0.06 / +0.09 | +0.64 / +0.85 / +0.56 | +0.03 / +0.60 / +0.07 | +0.83 | 0.43 |
| shared_free | 6 | +0.52 | +0.04 / +0.06 / +0.10 | +0.51 / +0.85 / +0.53 | +0.03 / +0.60 / +0.18 | +0.78 | 0.38 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.06 / -0.13 | +0.57 / +0.49 | +0.10 / +0.22 | +0.69 / +0.82 |
| free_free | -0.09 / -0.13 | +0.56 / +0.49 | +0.15 / +0.22 | +0.70 / +0.82 |
| none_free | +0.16 / -0.13 | +0.50 / +0.49 | -0.04 / +0.22 | +0.65 / +0.82 |
| sharedI_free | +0.15 / -0.13 | +0.42 / +0.49 | -0.02 / +0.22 | +0.70 / +0.82 |
| shared_free | +0.05 / -0.13 | +0.39 / +0.49 | -0.03 / +0.22 | +0.65 / +0.82 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +29.91 | -0.56 / +0.01 | +0.79 / +0.54 | +0.91 | 4 |
| free_free | +37.16 | -0.47 / +0.01 | +0.77 / +0.54 | +0.99 | 4 |
| none_free | +0.00 | +0.40 / -0.28 | +0.44 / +0.54 | +0.72 | 3 |
| sharedI_free | +30.95 | -0.05 / +0.01 | +0.56 / +0.54 | +0.81 | 4 |
| shared_free | -0.62 | -0.06 / -0.28 | +0.43 / +0.54 | +0.69 | 3 |
