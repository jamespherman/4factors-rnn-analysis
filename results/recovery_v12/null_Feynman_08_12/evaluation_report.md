# Context model evaluation (results\recovery_v12\null_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +14.15 | +0.03 / -0.03 / +0.05 | +0.58 / -0.08 / -0.08 | +0.08 / +0.21 / +0.46 | +0.83 | 0.25 |
| free_free | 3 | +7.05 | -0.00 / -0.03 / +0.37 | +0.61 / -0.08 / +0.11 | +0.09 / +0.21 / +0.72 | +0.87 | 0.18 |
| none_free | 3 | +0.00 | +0.01 / -0.03 / +0.33 | +0.02 / -0.08 / +0.20 | +0.02 / +0.21 / +0.42 | +0.21 | 0.17 |
| sharedI_free | 3 | +17.94 | +0.03 / -0.03 / -0.08 | +0.44 / -0.08 / -0.01 | +0.08 / +0.21 / +0.33 | +0.85 | 0.30 |
| shared_free | 3 | +2.54 | -0.13 / -0.03 / +0.03 | +0.40 / -0.08 / -0.16 | +0.02 / +0.21 / +0.77 | +0.58 | 0.22 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.07 / +0.03 | +0.19 / -0.25 | +0.37 / -0.05 | +0.43 / +0.72 |
| free_free | +0.05 / +0.03 | -0.14 / -0.25 | +0.13 / -0.05 | +0.48 / +0.72 |
| none_free | +0.21 / +0.03 | +0.08 / -0.25 | +0.39 / -0.05 | -0.08 / +0.72 |
| sharedI_free | -0.17 / +0.03 | +0.02 / -0.25 | +0.32 / -0.05 | +0.53 / +0.72 |
| shared_free | +0.30 / +0.03 | +0.13 / -0.25 | +0.13 / -0.05 | +0.25 / +0.72 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +25.49 | -0.35 / +0.24 | +0.43 / +0.10 | +0.81 | 1 |
| free_free | +5.30 | +0.11 / +0.24 | +0.11 / +0.10 | +0.82 | 1 |
| none_free | +0.00 | -0.13 / +0.24 | +0.74 / +0.10 | -0.58 | 1 |
| sharedI_free | +24.52 | -0.19 / +0.24 | +0.36 / +0.10 | +0.77 | 1 |
| shared_free | +7.01 | +0.25 / +0.24 | -0.14 / +0.10 | +0.81 | 1 |
