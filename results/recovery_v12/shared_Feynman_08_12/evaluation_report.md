# Context model evaluation (results\recovery_v12\shared_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +5.60 | +0.05 / +0.02 / +0.05 | +0.24 / +0.13 / -0.51 | +0.01 / +0.24 / +0.36 | +0.56 | 0.17 |
| free_free | 3 | +6.39 | +0.03 / +0.02 / +0.75 | +0.66 / +0.13 / -0.52 | +0.01 / +0.24 / +0.38 | +0.58 | 0.19 |
| none_free | 3 | +0.00 | +0.03 / +0.02 / -0.41 | +0.01 / +0.13 / +0.26 | +0.01 / +0.24 / +0.19 | +0.12 | 0.20 |
| sharedI_free | 3 | +0.65 | -0.03 / +0.02 / +0.92 | +0.13 / +0.13 / -0.50 | +0.04 / +0.24 / +0.20 | +0.60 | 0.10 |
| shared_free | 3 | +2.79 | -0.03 / +0.02 / +0.91 | +0.36 / +0.13 / -0.23 | +0.00 / +0.24 / +0.32 | +0.62 | 0.26 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.07 / +0.48 | -0.01 / +0.07 | +0.01 / -0.11 | +0.25 / +0.65 |
| free_free | +0.28 / +0.48 | -0.19 / +0.07 | -0.00 / -0.11 | +0.20 / +0.65 |
| none_free | -0.07 / +0.48 | +0.38 / +0.07 | -0.09 / -0.11 | -0.04 / +0.65 |
| sharedI_free | +0.49 / +0.48 | -0.15 / +0.07 | -0.20 / -0.11 | +0.34 / +0.65 |
| shared_free | +0.64 / +0.48 | -0.27 / +0.07 | -0.06 / -0.11 | +0.41 / +0.65 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.07 | +0.45 / +0.64 | +0.69 / -0.45 | +0.56 | 1 |
| free_free | +3.01 | +0.41 / +0.64 | +0.41 / -0.45 | +0.63 | 1 |
| none_free | +0.00 | -0.69 / +0.64 | +0.39 / -0.45 | +0.48 | 1 |
| sharedI_free | -2.24 | +0.57 / +0.64 | +0.31 / -0.45 | +0.56 | 1 |
| shared_free | +2.07 | +0.64 / +0.64 | -0.16 / -0.45 | +0.57 | 1 |
