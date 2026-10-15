# Context model evaluation (results\recovery_v12_drift\local_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -2.24 | -0.08 / +0.24 / +0.04 | +0.21 / +0.80 / +0.22 | +0.02 / +0.05 / -0.10 | -0.06 | 0.41 |
| free_free | 6 | -1.49 | -0.30 / +0.24 / +0.13 | +0.45 / +0.80 / +0.19 | +0.02 / +0.05 / -0.06 | +0.09 | 0.30 |
| none_free | 6 | +0.00 | -0.02 / +0.24 / -0.35 | -0.00 / +0.80 / -0.01 | +0.03 / +0.05 / -0.11 | +0.10 | 0.35 |
| sharedI_free | 6 | -2.16 | -0.09 / +0.24 / -0.09 | +0.12 / +0.80 / +0.22 | +0.02 / +0.05 / -0.04 | +0.03 | 0.30 |
| shared_free | 6 | -3.14 | -0.09 / +0.24 / -0.10 | +0.10 / +0.80 / +0.25 | +0.02 / +0.05 / -0.05 | +0.08 | 0.31 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.16 / +0.35 | +0.11 / +0.07 | -0.33 / +0.00 | -0.22 / -0.13 |
| free_free | -0.08 / +0.35 | +0.11 / +0.07 | -0.26 / +0.00 | -0.26 / -0.13 |
| none_free | -0.13 / +0.35 | +0.00 / +0.07 | -0.22 / +0.00 | -0.08 / -0.13 |
| sharedI_free | -0.33 / +0.35 | -0.05 / +0.07 | -0.39 / +0.00 | -0.25 / -0.13 |
| shared_free | -0.31 / +0.35 | -0.00 / +0.07 | -0.39 / +0.00 | -0.25 / -0.13 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -1.09 | -0.15 / +0.46 | +0.56 / +0.15 | +0.68 | 4 |
| free_free | -2.50 | +0.32 / +0.46 | +0.48 / +0.15 | -0.08 | 1 |
| none_free | +0.00 | +0.29 / +0.46 | -0.33 / +0.15 | +0.23 | 4 |
| sharedI_free | -0.72 | -0.29 / +0.46 | +0.44 / +0.15 | +0.40 | 4 |
| shared_free | -0.64 | -0.31 / +0.46 | +0.43 / +0.15 | +0.38 | 4 |
