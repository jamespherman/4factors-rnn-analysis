# Context model evaluation (results\recovery_v12_drift\local_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -5.29 | +0.41 / +0.93 / +0.46 | +0.85 / +1.16 / +0.59 | +0.03 / -0.10 / +0.03 | +0.58 | 0.48 |
| free_free | 6 | -5.68 | +0.68 / +0.93 / +0.61 | +1.41 / +1.16 / +0.73 | +0.04 / -0.10 / +0.11 | +0.65 | 0.42 |
| none_free | 6 | +0.00 | +0.03 / +0.93 / +0.10 | +0.09 / +1.16 / +0.38 | -0.01 / -0.10 / +0.05 | +0.48 | 0.46 |
| sharedI_free | 6 | -7.92 | +0.17 / +0.93 / +0.28 | +0.67 / +1.16 / +0.38 | +0.02 / -0.10 / +0.09 | +0.50 | 0.41 |
| shared_free | 6 | -9.01 | +0.21 / +0.93 / +0.30 | +0.88 / +1.16 / +0.42 | +0.00 / -0.10 / +0.12 | +0.60 | 0.45 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.18 / +0.17 | +0.31 / +0.34 | +0.09 / +0.47 | +0.34 / +0.63 |
| free_free | +0.24 / +0.17 | +0.39 / +0.34 | +0.21 / +0.47 | +0.43 / +0.63 |
| none_free | +0.20 / +0.17 | +0.15 / +0.34 | +0.10 / +0.47 | +0.34 / +0.63 |
| sharedI_free | +0.20 / +0.17 | +0.21 / +0.34 | +0.10 / +0.47 | +0.37 / +0.63 |
| shared_free | +0.23 / +0.17 | +0.26 / +0.34 | +0.21 / +0.47 | +0.47 / +0.63 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +3.54 | -0.09 / -0.13 | +0.28 / +0.30 | +0.84 | 4 |
| free_free | -13.86 | +0.04 / -0.13 | +0.30 / +0.30 | +0.72 | 4 |
| none_free | +0.00 | +0.01 / -0.13 | +0.25 / +0.30 | +0.79 | 4 |
| sharedI_free | +2.56 | -0.24 / -0.13 | +0.23 / +0.30 | +0.87 | 4 |
| shared_free | -2.73 | -0.23 / -0.13 | +0.26 / +0.30 | +0.89 | 4 |
