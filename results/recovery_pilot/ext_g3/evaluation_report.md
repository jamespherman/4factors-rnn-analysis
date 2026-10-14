# Context model evaluation (results\recovery_pilot\ext_g3)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +47.09 | +3.09 / +3.71 / +0.95 | +8.54 / +9.40 / +0.99 | +0.00 / +0.70 / -0.09 | +0.98 | 0.56 |
| free_free | 3 | +54.17 | +3.26 / +3.71 / +0.99 | +9.01 / +9.40 / +1.00 | +0.07 / +0.70 / -0.06 | +0.98 | 0.48 |
| none_free | 3 | +0.00 | -0.06 / +3.71 / -0.53 | +0.95 / +9.40 / +0.90 | +0.04 / +0.70 / +0.12 | +0.86 | 0.45 |
| sharedI_free | 3 | +38.83 | +2.68 / +3.71 / +0.95 | +7.14 / +9.40 / +0.98 | +0.06 / +0.70 / +0.12 | +0.98 | 0.56 |
| shared_free | 3 | +44.21 | +2.78 / +3.71 / +0.95 | +8.06 / +9.40 / +0.99 | +0.07 / +0.70 / +0.16 | +0.98 | 0.56 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.93 / +0.99 | +0.98 / +0.99 | +0.19 / +0.24 | +0.92 / +0.95 |
| free_free | +0.98 / +0.99 | +0.99 / +0.99 | +0.01 / +0.24 | +0.93 / +0.95 |
| none_free | -0.52 / +0.99 | +0.88 / +0.99 | +0.04 / +0.24 | +0.79 / +0.95 |
| sharedI_free | +0.93 / +0.99 | +0.97 / +0.99 | -0.02 / +0.24 | +0.93 / +0.95 |
| shared_free | +0.93 / +0.99 | +0.98 / +0.99 | -0.05 / +0.24 | +0.93 / +0.95 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +42.78 | +0.94 / +0.99 | +0.99 / +0.99 | +0.98 | 3 |
| free_free | +49.22 | +0.99 / +0.99 | +0.99 / +0.99 | +0.98 | 3 |
| none_free | +0.00 | -0.29 / +0.99 | +0.89 / +0.99 | +0.93 | 3 |
| sharedI_free | +34.84 | +0.93 / +0.99 | +0.97 / +0.99 | +0.98 | 3 |
| shared_free | +35.73 | +0.93 / +0.99 | +0.99 / +0.99 | +0.99 | 1 |
