# Context model evaluation (results\recovery_v12_drift\null_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -10.81 | -0.06 / +0.51 / +0.26 | +1.50 / +1.49 / +0.70 | +0.03 / +1.13 / +0.16 | +0.80 | 0.28 |
| free_free | 6 | -12.48 | +0.01 / +0.51 / +0.32 | +1.32 / +1.49 / +0.53 | +0.02 / +1.13 / +0.00 | +0.56 | 0.25 |
| none_free | 6 | +0.00 | -0.02 / +0.51 / +0.30 | +0.13 / +1.49 / -0.19 | +0.03 / +1.13 / +0.04 | +0.30 | 0.22 |
| sharedI_free | 6 | -10.59 | +0.00 / +0.51 / +0.16 | +1.57 / +1.49 / +0.73 | +0.02 / +1.13 / +0.13 | +0.90 | 0.24 |
| shared_free | 6 | -17.24 | -0.19 / +0.51 / -0.17 | +1.30 / +1.49 / +0.44 | +0.00 / +1.13 / -0.18 | +0.58 | 0.20 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.12 / -0.02 | +0.37 / +0.14 | +0.02 / +0.08 | +0.27 / +0.27 |
| free_free | +0.00 / -0.02 | +0.39 / +0.14 | +0.00 / +0.08 | +0.32 / +0.27 |
| none_free | -0.07 / -0.02 | +0.23 / +0.14 | +0.06 / +0.08 | +0.23 / +0.27 |
| sharedI_free | -0.30 / -0.02 | +0.41 / +0.14 | +0.00 / +0.08 | +0.27 / +0.27 |
| shared_free | +0.12 / -0.02 | +0.43 / +0.14 | +0.01 / +0.08 | +0.39 / +0.27 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -2.72 | -0.45 / -0.41 | -0.22 / -0.55 | +0.84 | 2 |
| free_free | +2.81 | +0.85 / +0.67 | +0.83 / +0.78 | +0.96 | 5 |
| none_free | +0.00 | +0.48 / +0.67 | +0.21 / +0.78 | +0.76 | 5 |
| sharedI_free | -9.43 | -0.55 / -0.41 | -0.33 / -0.55 | +0.87 | 2 |
| shared_free | -0.12 | +0.85 / +0.67 | +0.68 / +0.78 | +0.89 | 5 |
