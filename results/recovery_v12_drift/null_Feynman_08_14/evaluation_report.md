# Context model evaluation (results\recovery_v12_drift\null_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -1.84 | -0.25 / +0.03 / +0.36 | +0.87 / +1.06 / +0.75 | +0.02 / +0.45 / +0.49 | +0.74 | 0.33 |
| free_free | 6 | -2.69 | -0.20 / +0.03 / +0.19 | +1.19 / +1.06 / +0.52 | +0.03 / +0.45 / +0.42 | +0.70 | 0.24 |
| none_free | 6 | +0.00 | -0.02 / +0.03 / +0.52 | +0.01 / +1.06 / +0.04 | +0.03 / +0.45 / +0.44 | +0.65 | 0.32 |
| sharedI_free | 6 | -2.34 | -0.12 / +0.03 / +0.48 | +0.47 / +1.06 / +0.50 | +0.03 / +0.45 / +0.46 | +0.57 | 0.33 |
| shared_free | 6 | -1.24 | -0.16 / +0.03 / +0.47 | +0.46 / +1.06 / +0.50 | +0.02 / +0.45 / +0.47 | +0.58 | 0.31 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.07 / +0.12 | +0.52 / +0.57 | +0.33 / +0.05 | +0.48 / +0.49 |
| free_free | -0.03 / +0.12 | +0.39 / +0.57 | +0.25 / +0.05 | +0.44 / +0.49 |
| none_free | +0.25 / +0.12 | -0.08 / +0.57 | +0.03 / +0.05 | +0.40 / +0.49 |
| sharedI_free | +0.22 / +0.12 | +0.37 / +0.57 | +0.36 / +0.05 | +0.41 / +0.49 |
| shared_free | +0.19 / +0.12 | +0.37 / +0.57 | +0.40 / +0.05 | +0.43 / +0.49 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -2.75 | +0.56 / +0.37 | +0.76 / +0.72 | +0.84 | 1 |
| free_free | -1.34 | -0.02 / +0.37 | +0.72 / +0.72 | +0.89 | 1 |
| none_free | +0.00 | +0.85 / +0.37 | +0.26 / +0.72 | +0.79 | 4 |
| sharedI_free | -3.19 | +0.87 / +0.37 | +0.76 / +0.72 | +0.56 | 1 |
| shared_free | -1.76 | +0.87 / +0.37 | +0.77 / +0.72 | +0.60 | 1 |
