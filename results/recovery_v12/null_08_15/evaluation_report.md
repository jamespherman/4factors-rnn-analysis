# Context model evaluation (results\recovery_v12\null_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +6.86 | +0.26 / +0.01 / +0.02 | +0.47 / -0.02 / +0.19 | +0.12 / -0.03 / +0.12 | +0.78 | 0.37 |
| free_free | 3 | +3.05 | +0.24 / +0.01 / +0.19 | +0.45 / -0.02 / +0.30 | -0.05 / -0.03 / +0.21 | +0.71 | 0.31 |
| none_free | 3 | +0.00 | +0.08 / +0.01 / -0.04 | +0.07 / -0.02 / +0.39 | -0.04 / -0.03 / +0.22 | +0.88 | 0.39 |
| sharedI_free | 3 | +4.67 | +0.18 / +0.01 / -0.05 | +0.63 / -0.02 / +0.34 | -0.01 / -0.03 / +0.22 | +0.87 | 0.39 |
| shared_free | 3 | +6.61 | +0.04 / +0.01 / +0.06 | +0.45 / -0.02 / +0.33 | -0.00 / -0.03 / +0.26 | +0.91 | 0.34 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.25 / +0.07 | +0.08 / -0.05 | -0.00 / +0.11 | +0.79 / +0.99 |
| free_free | -0.20 / +0.07 | +0.14 / -0.05 | +0.10 / +0.11 | +0.72 / +0.99 |
| none_free | -0.32 / +0.07 | +0.12 / -0.05 | +0.04 / +0.11 | +0.87 / +0.99 |
| sharedI_free | -0.25 / +0.07 | -0.03 / -0.05 | +0.12 / +0.11 | +0.86 / +0.99 |
| shared_free | -0.17 / +0.07 | -0.06 / -0.05 | +0.05 / +0.11 | +0.90 / +0.99 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.31 | -0.18 / +0.05 | +0.25 / -0.24 | +1.00 | 1 |
| free_free | -2.06 | -0.17 / +0.05 | +0.39 / -0.24 | +1.00 | 1 |
| none_free | +0.00 | -0.42 / +0.05 | +0.30 / -0.24 | +1.00 | 1 |
| sharedI_free | +0.92 | -0.37 / +0.05 | -0.19 / -0.24 | +1.00 | 1 |
| shared_free | -2.27 | -0.37 / +0.05 | -0.28 / -0.24 | +1.00 | 1 |
