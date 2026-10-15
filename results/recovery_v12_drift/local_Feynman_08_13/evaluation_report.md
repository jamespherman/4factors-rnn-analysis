# Context model evaluation (results\recovery_v12_drift\local_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +0.22 | +0.04 / +0.07 / +0.61 | +1.14 / +0.62 / +0.82 | +0.03 / +0.03 / +0.03 | +0.85 | 0.42 |
| free_free | 6 | +0.43 | +0.06 / +0.07 / +0.62 | +1.24 / +0.62 / +0.84 | +0.02 / +0.03 / +0.04 | +0.87 | 0.30 |
| none_free | 6 | +0.00 | +0.05 / +0.07 / -0.10 | +0.09 / +0.62 / +0.71 | -0.00 / +0.03 / -0.05 | +0.75 | 0.43 |
| sharedI_free | 6 | +0.16 | -0.07 / +0.07 / +0.23 | +0.42 / +0.62 / +0.49 | -0.00 / +0.03 / -0.04 | +0.80 | 0.43 |
| shared_free | 6 | -0.23 | -0.07 / +0.07 / +0.24 | +0.39 / +0.62 / +0.50 | +0.01 / +0.03 / -0.08 | +0.80 | 0.38 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.13 / -0.09 | +0.63 / +0.61 | +0.08 / +0.04 | +0.79 / +0.89 |
| free_free | +0.06 / -0.09 | +0.65 / +0.61 | +0.08 / +0.04 | +0.79 / +0.89 |
| none_free | +0.13 / -0.09 | +0.58 / +0.61 | +0.08 / +0.04 | +0.69 / +0.89 |
| sharedI_free | -0.02 / -0.09 | +0.39 / +0.61 | +0.06 / +0.04 | +0.74 / +0.89 |
| shared_free | +0.00 / -0.09 | +0.37 / +0.61 | +0.08 / +0.04 | +0.75 / +0.89 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.84 | +0.42 / -0.03 | +0.83 / +0.75 | +0.84 | 4 |
| free_free | +4.00 | +0.17 / -0.03 | +0.84 / +0.75 | +0.84 | 4 |
| none_free | +0.00 | -0.74 / -0.03 | +0.80 / +0.75 | +0.73 | 4 |
| sharedI_free | +0.36 | +0.27 / -0.03 | +0.55 / +0.75 | +0.77 | 4 |
| shared_free | -1.97 | +0.23 / -0.03 | +0.48 / +0.75 | +0.73 | 4 |
