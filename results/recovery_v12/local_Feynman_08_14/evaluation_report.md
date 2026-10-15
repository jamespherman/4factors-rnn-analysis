# Context model evaluation (results\recovery_v12\local_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +0.47 | +0.08 / +0.13 / +0.75 | +0.84 / +0.96 / +0.88 | +0.02 / -0.01 / -0.06 | +0.82 | 0.46 |
| free_free | 3 | +0.49 | -0.03 / +0.13 / +0.83 | +0.95 / +0.96 / +0.90 | +0.01 / -0.01 / +0.05 | +0.82 | 0.29 |
| none_free | 3 | +0.00 | +0.02 / +0.13 / -0.61 | +0.07 / +0.96 / +0.58 | +0.01 / -0.01 / -0.30 | +0.76 | 0.38 |
| sharedI_free | 3 | -0.22 | -0.01 / +0.13 / +0.43 | +0.13 / +0.96 / +0.34 | +0.01 / -0.01 / -0.28 | +0.80 | 0.29 |
| shared_free | 3 | +0.24 | +0.03 / +0.13 / +0.39 | +0.16 / +0.96 / +0.34 | +0.00 / -0.01 / -0.10 | +0.75 | 0.39 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.42 / +0.54 | +0.70 / +0.60 | +0.17 / +0.02 | +0.27 / +0.10 |
| free_free | +0.49 / +0.54 | +0.73 / +0.60 | +0.32 / +0.02 | +0.26 / +0.10 |
| none_free | -0.44 / +0.54 | +0.18 / +0.60 | +0.10 / +0.02 | +0.28 / +0.10 |
| sharedI_free | +0.33 / +0.54 | +0.01 / +0.60 | -0.01 / +0.02 | +0.29 / +0.10 |
| shared_free | +0.31 / +0.54 | +0.06 / +0.60 | +0.16 / +0.02 | +0.27 / +0.10 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.47 | +0.16 / +0.11 | +0.78 / +0.73 | +0.84 | 3 |
| free_free | +0.39 | +0.19 / +0.11 | +0.74 / +0.73 | +0.83 | 3 |
| none_free | +0.00 | +0.12 / +0.11 | +0.17 / +0.73 | +0.75 | 3 |
| sharedI_free | -0.72 | +0.22 / +0.11 | +0.01 / +0.73 | +0.87 | 3 |
| shared_free | +0.04 | +0.21 / +0.11 | -0.06 / +0.73 | +0.72 | 3 |
