# Context model evaluation (results\recovery_v12_drift\shared_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -5.71 | +0.02 / +0.50 / +0.34 | +0.95 / +1.43 / +0.36 | +0.02 / +0.93 / +0.19 | +0.65 | 0.32 |
| free_free | 6 | -17.83 | -0.09 / +0.50 / +0.13 | +1.48 / +1.43 / +0.44 | +0.05 / +0.93 / +0.07 | +0.68 | 0.19 |
| none_free | 6 | +0.00 | -0.01 / +0.50 / +0.22 | +0.00 / +1.43 / -0.29 | -0.01 / +0.93 / +0.15 | +0.21 | 0.22 |
| sharedI_free | 6 | -9.16 | -0.06 / +0.50 / +0.06 | +0.89 / +1.43 / +0.29 | -0.00 / +0.93 / +0.10 | +0.87 | 0.12 |
| shared_free | 6 | -16.72 | -0.11 / +0.50 / -0.13 | +1.47 / +1.43 / +0.32 | +0.02 / +0.93 / +0.12 | +0.63 | 0.19 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.15 / -0.09 | +0.22 / +0.05 | +0.15 / +0.22 | +0.21 / +0.44 |
| free_free | +0.14 / -0.09 | +0.27 / +0.05 | -0.05 / +0.22 | +0.38 / +0.44 |
| none_free | -0.25 / -0.09 | -0.22 / +0.05 | +0.05 / +0.22 | +0.25 / +0.44 |
| sharedI_free | -0.43 / -0.09 | +0.16 / +0.05 | +0.10 / +0.22 | +0.40 / +0.44 |
| shared_free | +0.21 / -0.09 | +0.33 / +0.05 | +0.02 / +0.22 | +0.47 / +0.44 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.43 | +0.11 / -0.41 | -0.29 / -0.16 | +0.65 | 2 |
| free_free | +2.31 | +0.40 / +0.35 | +0.38 / +0.60 | +0.95 | 5 |
| none_free | +0.00 | +0.31 / -0.41 | +0.27 / -0.16 | +0.05 | 2 |
| sharedI_free | -5.76 | -0.57 / -0.41 | -0.45 / -0.16 | +0.80 | 2 |
| shared_free | -7.26 | +0.42 / +0.35 | +0.12 / +0.60 | +0.90 | 5 |
