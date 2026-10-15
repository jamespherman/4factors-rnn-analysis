# Context model evaluation (results\recovery_v12\ext_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +0.99 | +0.48 / +0.41 / +0.80 | +0.63 / +0.13 / +0.72 | -0.01 / -0.74 / +0.01 | +0.39 | 0.44 |
| free_free | 3 | +2.58 | +0.54 / +0.41 / +0.85 | +0.64 / +0.13 / +0.73 | -0.03 / -0.74 / +0.13 | +0.49 | 0.35 |
| none_free | 3 | +0.00 | +0.14 / +0.41 / +0.30 | -0.02 / +0.13 / +0.62 | -0.11 / -0.74 / +0.24 | +0.56 | 0.41 |
| sharedI_free | 3 | +0.09 | +0.21 / +0.41 / +0.42 | +0.19 / +0.13 / +0.54 | -0.08 / -0.74 / +0.17 | +0.48 | 0.46 |
| shared_free | 3 | +0.85 | +0.06 / +0.41 / +0.30 | -0.07 / +0.13 / +0.55 | -0.03 / -0.74 / -0.06 | +0.34 | 0.45 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.38 / +0.43 | +0.41 / +0.54 | -0.06 / +0.24 | +0.19 / +0.77 |
| free_free | +0.40 / +0.43 | +0.41 / +0.54 | -0.06 / +0.24 | +0.35 / +0.77 |
| none_free | +0.20 / +0.43 | +0.36 / +0.54 | +0.14 / +0.24 | +0.43 / +0.77 |
| sharedI_free | +0.01 / +0.43 | +0.21 / +0.54 | +0.12 / +0.24 | +0.29 / +0.77 |
| shared_free | -0.02 / +0.43 | +0.23 / +0.54 | -0.08 / +0.24 | +0.12 / +0.77 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.76 | +0.37 / +0.40 | +0.44 / +0.57 | +0.48 | 1 |
| free_free | +6.03 | +0.40 / +0.40 | +0.49 / +0.57 | +0.86 | 1 |
| none_free | +0.00 | +0.23 / +0.40 | +0.48 / +0.57 | +0.56 | 1 |
| sharedI_free | +0.19 | -0.24 / +0.40 | +0.27 / +0.57 | +0.47 | 1 |
| shared_free | +0.95 | -0.23 / +0.40 | +0.24 / +0.57 | +0.48 | 1 |
