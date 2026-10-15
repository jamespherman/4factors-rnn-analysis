# Context model evaluation (results\recovery_v12\null_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -7.30 | +0.13 / +0.04 / +0.16 | +0.50 / -0.08 / +0.50 | -0.06 / +0.02 / -0.28 | +0.72 | 0.34 |
| free_free | 3 | -4.36 | +0.14 / +0.04 / +0.26 | +0.74 / -0.08 / +0.57 | -0.03 / +0.02 / -0.11 | +0.78 | 0.39 |
| none_free | 3 | +0.00 | +0.01 / +0.04 / -0.06 | +0.07 / -0.08 / +0.52 | -0.04 / +0.02 / -0.24 | +0.73 | 0.37 |
| sharedI_free | 3 | -1.23 | +0.00 / +0.04 / -0.04 | +0.00 / -0.08 / +0.30 | -0.05 / +0.02 / -0.30 | +0.63 | 0.39 |
| shared_free | 3 | -2.05 | -0.00 / +0.04 / -0.02 | +0.01 / -0.08 / +0.26 | -0.07 / +0.02 / -0.31 | +0.61 | 0.39 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.00 / -0.17 | +0.16 / +0.03 | -0.11 / -0.02 | +0.61 / +0.91 |
| free_free | +0.02 / -0.17 | +0.05 / +0.03 | -0.05 / -0.02 | +0.70 / +0.91 |
| none_free | +0.18 / -0.17 | +0.03 / +0.03 | -0.30 / -0.02 | +0.64 / +0.91 |
| sharedI_free | +0.13 / -0.17 | +0.10 / +0.03 | -0.30 / -0.02 | +0.54 / +0.91 |
| shared_free | +0.09 / -0.17 | +0.08 / +0.03 | -0.25 / -0.02 | +0.53 / +0.91 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -2.99 | -0.05 / -0.40 | +0.25 / +0.19 | +0.80 | 1 |
| free_free | -3.90 | -0.12 / -0.40 | +0.18 / +0.19 | +0.86 | 1 |
| none_free | +0.00 | +0.39 / -0.40 | +0.12 / +0.19 | +0.74 | 1 |
| sharedI_free | -0.30 | +0.48 / -0.40 | +0.12 / +0.19 | +0.68 | 1 |
| shared_free | -1.05 | +0.47 / -0.40 | +0.14 / +0.19 | +0.65 | 1 |
