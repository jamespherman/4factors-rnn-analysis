# Context model evaluation (results\recovery_v12_drift\null_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +1.33 | +0.21 / +0.07 / +0.40 | +1.26 / +0.03 / +0.53 | +0.21 / +0.60 / +0.47 | +0.85 | 0.45 |
| free_free | 6 | +2.37 | +0.17 / +0.07 / +0.45 | +1.24 / +0.03 / +0.56 | +0.26 / +0.60 / +0.54 | +0.86 | 0.31 |
| none_free | 6 | +0.00 | -0.00 / +0.07 / +0.28 | +0.02 / +0.03 / +0.39 | +0.03 / +0.60 / +0.37 | +0.67 | 0.39 |
| sharedI_free | 6 | +3.44 | +0.05 / +0.07 / -0.06 | +0.90 / +0.03 / +0.53 | +0.12 / +0.60 / +0.51 | +0.79 | 0.38 |
| shared_free | 6 | +4.35 | +0.07 / +0.07 / -0.02 | +1.12 / +0.03 / +0.52 | +0.10 / +0.60 / +0.51 | +0.84 | 0.38 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.02 / -0.08 | +0.13 / -0.05 | +0.23 / +0.15 | +0.79 / +0.92 |
| free_free | -0.07 / -0.08 | +0.21 / -0.05 | +0.29 / +0.15 | +0.79 / +0.92 |
| none_free | +0.23 / -0.08 | -0.06 / -0.05 | +0.17 / +0.15 | +0.61 / +0.92 |
| sharedI_free | -0.12 / -0.08 | +0.11 / -0.05 | +0.22 / +0.15 | +0.75 / +0.92 |
| shared_free | -0.08 / -0.08 | +0.13 / -0.05 | +0.26 / +0.15 | +0.79 / +0.92 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +2.60 | +0.12 / +0.25 | +0.47 / +0.22 | +0.96 | 5 |
| free_free | +2.52 | -0.04 / +0.25 | +0.59 / +0.22 | +0.99 | 5 |
| none_free | +0.00 | +0.56 / +0.25 | -0.23 / +0.22 | +0.95 | 5 |
| sharedI_free | +3.32 | -0.32 / +0.25 | +0.52 / +0.22 | +0.97 | 5 |
| shared_free | +3.88 | -0.45 / +0.25 | +0.53 / +0.22 | +0.96 | 5 |
