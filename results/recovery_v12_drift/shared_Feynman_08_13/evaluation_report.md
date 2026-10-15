# Context model evaluation (results\recovery_v12_drift\shared_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -1.42 | -0.04 / -0.30 / +0.55 | +1.78 / +1.02 / +0.78 | +0.01 / +0.22 / +0.00 | +0.83 | 0.46 |
| free_free | 6 | -0.62 | -0.01 / -0.30 / +0.64 | +2.00 / +1.02 / +0.85 | +0.06 / +0.22 / +0.14 | +0.88 | 0.34 |
| none_free | 6 | +0.00 | +0.08 / -0.30 / -0.17 | +0.15 / +1.02 / +0.59 | -0.03 / +0.22 / -0.16 | +0.70 | 0.46 |
| sharedI_free | 6 | -0.34 | -0.13 / -0.30 / +0.53 | +0.63 / +1.02 / +0.47 | -0.00 / +0.22 / -0.05 | +0.77 | 0.44 |
| shared_free | 6 | +0.29 | -0.21 / -0.30 / +0.50 | +0.78 / +1.02 / +0.50 | -0.01 / +0.22 / -0.02 | +0.79 | 0.43 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.12 / +0.17 | +0.51 / +0.46 | +0.10 / +0.29 | +0.68 / +0.84 |
| free_free | +0.23 / +0.17 | +0.55 / +0.46 | +0.18 / +0.29 | +0.74 / +0.84 |
| none_free | +0.13 / +0.17 | +0.47 / +0.46 | -0.01 / +0.29 | +0.57 / +0.84 |
| sharedI_free | +0.23 / +0.17 | +0.33 / +0.46 | +0.10 / +0.29 | +0.66 / +0.84 |
| shared_free | +0.16 / +0.17 | +0.34 / +0.46 | +0.10 / +0.29 | +0.67 / +0.84 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +29.38 | -0.01 / +0.32 | +0.84 / +0.57 | +0.93 | 4 |
| free_free | +29.94 | -0.07 / +0.32 | +0.83 / +0.57 | +0.91 | 4 |
| none_free | +0.00 | +0.08 / +0.16 | +0.13 / +0.35 | +0.66 | 3 |
| sharedI_free | -1.82 | -0.05 / +0.16 | +0.32 / +0.35 | +0.66 | 3 |
| shared_free | -3.82 | -0.08 / +0.16 | +0.31 / +0.35 | +0.67 | 3 |
