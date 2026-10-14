# Context model evaluation (results\recovery_pilot\shared_g3)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +1.31 | -0.03 / +0.04 / +0.95 | +0.14 / +0.07 / +0.61 | -0.02 / -0.26 / +0.05 | +0.50 | 0.41 |
| free_free | 3 | +2.10 | -0.01 / +0.04 / +0.94 | +0.57 / +0.07 / +0.70 | -0.02 / -0.26 / -0.29 | +0.51 | 0.38 |
| none_free | 3 | +0.00 | +0.03 / +0.04 / +0.84 | -0.04 / +0.07 / +0.40 | +0.05 / -0.26 / +0.12 | +0.45 | 0.40 |
| sharedI_free | 3 | +1.73 | -0.15 / +0.04 / +0.92 | +0.09 / +0.07 / +0.55 | +0.01 / -0.26 / -0.06 | +0.37 | 0.42 |
| shared_free | 3 | +1.94 | -0.12 / +0.04 / +0.93 | +0.17 / +0.07 / +0.56 | -0.01 / -0.26 / -0.05 | +0.54 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.75 / +0.71 | +0.19 / +0.21 | -0.11 / +0.41 | +0.19 / +0.32 |
| free_free | +0.72 / +0.71 | +0.30 / +0.21 | -0.38 / +0.41 | +0.15 / +0.32 |
| none_free | +0.72 / +0.71 | +0.18 / +0.21 | +0.06 / +0.41 | +0.13 / +0.32 |
| sharedI_free | +0.79 / +0.71 | +0.17 / +0.21 | -0.16 / +0.41 | +0.15 / +0.32 |
| shared_free | +0.79 / +0.71 | +0.17 / +0.21 | -0.22 / +0.41 | +0.15 / +0.32 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +2.67 | +0.76 / +0.71 | +0.20 / +0.21 | +0.75 | 3 |
| free_free | +1.53 | +0.70 / +0.71 | +0.25 / +0.21 | +0.34 | 1 |
| none_free | +0.00 | +0.69 / +0.71 | +0.24 / +0.21 | +0.35 | 2 |
| sharedI_free | +2.19 | +0.79 / +0.71 | +0.18 / +0.21 | +0.48 | 2 |
| shared_free | +3.89 | +0.79 / +0.71 | +0.17 / +0.21 | +0.77 | 3 |
