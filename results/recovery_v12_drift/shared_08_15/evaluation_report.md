# Context model evaluation (results\recovery_v12_drift\shared_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -47.12 | +0.19 / +0.26 / +0.17 | +2.23 / +1.10 / +0.21 | +0.10 / +3.76 / +0.18 | +0.51 | 0.38 |
| free_free | 6 | -7.30 | +0.27 / +0.26 / +0.43 | +2.79 / +1.10 / +0.40 | +1.17 / +3.76 / +0.55 | +0.66 | 0.35 |
| none_free | 6 | +0.00 | +0.03 / +0.26 / +0.05 | +0.06 / +1.10 / +0.17 | +0.03 / +3.76 / +0.15 | +0.22 | 0.43 |
| sharedI_free | 6 | -3.24 | +0.27 / +0.26 / +0.09 | +2.81 / +1.10 / +0.15 | +0.42 / +3.76 / +0.34 | +0.55 | 0.40 |
| shared_free | 6 | +9.76 | +0.25 / +0.26 / +0.01 | +3.54 / +1.10 / +0.19 | +0.66 / +3.76 / +0.22 | +0.63 | 0.40 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.14 / +0.10 | +0.17 / +0.26 | +0.11 / +0.58 | +0.47 / +0.82 |
| free_free | +0.11 / +0.10 | +0.28 / +0.26 | +0.34 / +0.58 | +0.59 / +0.82 |
| none_free | -0.09 / +0.10 | -0.01 / +0.26 | +0.07 / +0.58 | +0.19 / +0.82 |
| sharedI_free | -0.03 / +0.10 | +0.13 / +0.26 | +0.20 / +0.58 | +0.50 / +0.82 |
| shared_free | +0.04 / +0.10 | +0.13 / +0.26 | +0.06 / +0.58 | +0.57 / +0.82 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -21.06 | -0.11 / +0.39 | +0.07 / +0.29 | +0.46 | 4 |
| free_free | +26.20 | +0.32 / +0.39 | +0.51 / +0.29 | +0.94 | 4 |
| none_free | +0.00 | -0.15 / +0.39 | +0.18 / +0.29 | +0.73 | 4 |
| sharedI_free | +14.57 | -0.28 / +0.39 | +0.18 / +0.29 | +0.91 | 4 |
| shared_free | +16.01 | -0.22 / +0.39 | +0.19 / +0.29 | +0.87 | 4 |
