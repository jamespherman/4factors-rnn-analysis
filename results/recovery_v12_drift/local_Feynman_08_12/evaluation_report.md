# Context model evaluation (results\recovery_v12_drift\local_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +0.04 | +0.12 / +0.67 / +0.26 | +1.19 / +2.13 / +0.70 | +0.05 / +1.64 / +0.11 | +0.70 | 0.33 |
| free_free | 6 | -5.14 | -0.29 / +0.67 / -0.16 | +1.78 / +2.13 / +0.65 | +0.03 / +1.64 / -0.10 | +0.59 | 0.19 |
| none_free | 6 | +0.00 | +0.03 / +0.67 / -0.06 | -0.02 / +2.13 / -0.50 | +0.05 / +1.64 / +0.14 | +0.18 | 0.20 |
| sharedI_free | 6 | -0.08 | +0.41 / +0.67 / +0.03 | +1.48 / +2.13 / +0.76 | +0.04 / +1.64 / +0.07 | +0.85 | 0.16 |
| shared_free | 6 | -9.33 | +0.58 / +0.67 / -0.21 | +2.82 / +2.13 / +0.59 | +0.04 / +1.64 / -0.01 | +0.61 | 0.23 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.28 / +0.33 | -0.05 / +0.11 | +0.10 / +0.23 | +0.23 / +0.45 |
| free_free | -0.06 / +0.33 | +0.17 / +0.11 | +0.02 / +0.23 | +0.29 / +0.45 |
| none_free | -0.00 / +0.33 | +0.02 / +0.11 | -0.09 / +0.23 | +0.36 / +0.45 |
| sharedI_free | +0.02 / +0.33 | +0.08 / +0.11 | +0.07 / +0.23 | +0.38 / +0.45 |
| shared_free | +0.04 / +0.33 | +0.15 / +0.11 | +0.11 / +0.23 | +0.33 / +0.45 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +24.38 | +0.60 / +0.21 | -0.41 / -0.35 | +0.83 | 2 |
| free_free | +14.39 | -0.56 / +0.94 | +0.69 / +0.64 | +0.80 | 5 |
| none_free | +0.00 | -0.59 / +0.94 | -0.18 / +0.64 | +0.50 | 5 |
| sharedI_free | +21.79 | +0.44 / +0.21 | -0.29 / -0.35 | +0.89 | 2 |
| shared_free | -12.64 | +0.73 / +0.94 | +0.72 / +0.64 | +0.96 | 5 |
