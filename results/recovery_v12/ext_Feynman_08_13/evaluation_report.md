# Context model evaluation (results\recovery_v12\ext_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -1.16 | +0.14 / +0.16 / +0.37 | +0.48 / +0.30 / +0.72 | -0.08 / +0.01 / +0.09 | +0.78 | 0.54 |
| free_free | 3 | -2.28 | +0.22 / +0.16 / +0.36 | +0.63 / +0.30 / +0.73 | -0.06 / +0.01 / +0.11 | +0.79 | 0.33 |
| none_free | 3 | +0.00 | +0.02 / +0.16 / -0.05 | +0.06 / +0.30 / +0.46 | -0.07 / +0.01 / +0.11 | +0.64 | 0.42 |
| sharedI_free | 3 | +0.23 | -0.01 / +0.16 / -0.16 | +0.07 / +0.30 / +0.44 | -0.06 / +0.01 / +0.12 | +0.69 | 0.35 |
| shared_free | 3 | +0.28 | -0.02 / +0.16 / -0.11 | +0.07 / +0.30 / +0.42 | -0.07 / +0.01 / +0.13 | +0.64 | 0.35 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.14 / +0.20 | +0.44 / +0.53 | -0.08 / +0.09 | +0.70 / +0.88 |
| free_free | +0.18 / +0.20 | +0.48 / +0.53 | -0.10 / +0.09 | +0.71 / +0.88 |
| none_free | +0.08 / +0.20 | +0.20 / +0.53 | -0.04 / +0.09 | +0.57 / +0.88 |
| sharedI_free | -0.19 / +0.20 | +0.27 / +0.53 | -0.06 / +0.09 | +0.62 / +0.88 |
| shared_free | -0.18 / +0.20 | +0.25 / +0.53 | -0.05 / +0.09 | +0.57 / +0.88 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -3.71 | +0.22 / +0.36 | +0.47 / +0.33 | +0.86 | 1 |
| free_free | -4.14 | +0.22 / +0.36 | +0.53 / +0.33 | +0.86 | 1 |
| none_free | +0.00 | +0.24 / -0.03 | +0.38 / +0.65 | +0.60 | 2 |
| sharedI_free | -0.16 | -0.10 / -0.03 | +0.32 / +0.65 | +0.68 | 2 |
| shared_free | -0.80 | -0.20 / -0.03 | +0.26 / +0.65 | +0.57 | 2 |
