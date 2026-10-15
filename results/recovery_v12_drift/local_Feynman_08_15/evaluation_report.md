# Context model evaluation (results\recovery_v12_drift\local_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +1.61 | +0.19 / +0.19 / +0.66 | +0.74 / +0.11 / +0.64 | +0.14 / +0.08 / +0.22 | +0.76 | 0.46 |
| free_free | 6 | +0.73 | +0.16 / +0.19 / +0.64 | +0.76 / +0.11 / +0.66 | +0.18 / +0.08 / +0.22 | +0.76 | 0.36 |
| none_free | 6 | +0.00 | +0.01 / +0.19 / -0.01 | +0.02 / +0.11 / +0.20 | +0.02 / +0.08 / +0.11 | +0.55 | 0.40 |
| sharedI_free | 6 | +3.25 | +0.05 / +0.19 / +0.07 | +0.50 / +0.11 / +0.46 | +0.16 / +0.08 / +0.25 | +0.83 | 0.38 |
| shared_free | 6 | +0.80 | +0.06 / +0.19 / +0.12 | +0.29 / +0.11 / +0.36 | +0.04 / +0.08 / +0.23 | +0.71 | 0.31 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.37 / +0.53 | +0.43 / +0.25 | -0.07 / +0.13 | +0.73 / +0.93 |
| free_free | +0.35 / +0.53 | +0.36 / +0.25 | -0.08 / +0.13 | +0.72 / +0.93 |
| none_free | +0.07 / +0.53 | +0.03 / +0.25 | -0.04 / +0.13 | +0.54 / +0.93 |
| sharedI_free | -0.09 / +0.53 | +0.16 / +0.25 | -0.04 / +0.13 | +0.79 / +0.93 |
| shared_free | -0.02 / +0.53 | +0.30 / +0.25 | -0.01 / +0.13 | +0.70 / +0.93 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.63 | +0.63 / +0.73 | +0.51 / +0.28 | +0.99 | 5 |
| free_free | +2.06 | +0.59 / +0.73 | +0.51 / +0.28 | +0.99 | 5 |
| none_free | +0.00 | +0.52 / +0.73 | -0.08 / +0.28 | +0.99 | 5 |
| sharedI_free | +2.04 | -0.20 / +0.73 | +0.43 / +0.28 | +0.99 | 5 |
| shared_free | +2.26 | -0.06 / +0.73 | +0.45 / +0.28 | +0.98 | 5 |
