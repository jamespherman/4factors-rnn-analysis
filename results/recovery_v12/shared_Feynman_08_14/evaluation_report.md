# Context model evaluation (results\recovery_v12\shared_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -0.68 | +0.11 / +0.15 / +0.91 | +0.10 / +0.26 / +0.42 | +0.01 / +0.20 / -0.16 | +0.66 | 0.32 |
| free_free | 3 | -0.51 | +0.13 / +0.15 / +0.88 | +0.21 / +0.26 / +0.55 | +0.01 / +0.20 / -0.22 | +0.64 | 0.34 |
| none_free | 3 | +0.00 | +0.00 / +0.15 / +0.05 | +0.03 / +0.26 / +0.17 | +0.01 / +0.20 / -0.12 | +0.66 | 0.32 |
| sharedI_free | 3 | -0.08 | +0.16 / +0.15 / +0.89 | +0.11 / +0.26 / +0.29 | -0.00 / +0.20 / -0.41 | +0.58 | 0.31 |
| shared_free | 3 | -0.68 | +0.14 / +0.15 / +0.88 | +0.13 / +0.26 / +0.19 | +0.01 / +0.20 / -0.19 | +0.58 | 0.36 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.49 / +0.47 | -0.16 / +0.16 | -0.15 / +0.23 | +0.12 / -0.15 |
| free_free | +0.46 / +0.47 | -0.02 / +0.16 | -0.19 / +0.23 | +0.12 / -0.15 |
| none_free | -0.07 / +0.47 | +0.13 / +0.16 | -0.06 / +0.23 | +0.10 / -0.15 |
| sharedI_free | +0.52 / +0.47 | -0.25 / +0.16 | -0.32 / +0.23 | +0.01 / -0.15 |
| shared_free | +0.53 / +0.47 | -0.37 / +0.16 | -0.14 / +0.23 | +0.02 / -0.15 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -1.48 | +0.23 / +0.18 | -0.34 / +0.70 | +0.59 | 3 |
| free_free | -0.62 | +0.31 / +0.18 | +0.22 / +0.70 | +0.64 | 3 |
| none_free | +0.00 | -0.06 / +0.18 | -0.15 / +0.70 | +0.61 | 3 |
| sharedI_free | -1.14 | +0.32 / +0.18 | -0.71 / +0.70 | +0.55 | 3 |
| shared_free | -1.77 | +0.29 / +0.18 | -0.70 / +0.70 | +0.51 | 3 |
