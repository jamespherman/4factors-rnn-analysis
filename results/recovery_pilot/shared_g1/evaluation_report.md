# Context model evaluation (results\recovery_pilot\shared_g1)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -0.59 | -0.14 / -0.15 / +0.83 | +0.05 / -0.05 / +0.68 | +0.04 / -0.10 / -0.14 | +0.15 | 0.41 |
| free_free | 3 | -2.63 | -0.04 / -0.15 / +0.82 | +0.43 / -0.05 / +0.61 | +0.03 / -0.10 / -0.13 | +0.16 | 0.38 |
| none_free | 3 | +0.00 | +0.02 / -0.15 / +0.18 | +0.03 / -0.05 / +0.52 | +0.02 / -0.10 / -0.14 | +0.32 | 0.30 |
| sharedI_free | 3 | +0.56 | -0.08 / -0.15 / +0.80 | +0.09 / -0.05 / +0.54 | +0.03 / -0.10 / -0.10 | +0.26 | 0.40 |
| shared_free | 3 | +0.68 | -0.10 / -0.15 / +0.81 | +0.08 / -0.05 / +0.50 | +0.01 / -0.10 / -0.15 | +0.32 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.47 / +0.19 | +0.28 / +0.31 | +0.07 / -0.06 | +0.08 / +0.51 |
| free_free | +0.53 / +0.19 | +0.25 / +0.31 | +0.06 / -0.06 | -0.01 / +0.51 |
| none_free | +0.17 / +0.19 | +0.13 / +0.31 | +0.01 / -0.06 | +0.04 / +0.51 |
| sharedI_free | +0.38 / +0.19 | +0.06 / +0.31 | +0.11 / -0.06 | -0.01 / +0.51 |
| shared_free | +0.38 / +0.19 | +0.04 / +0.31 | -0.05 / -0.06 | +0.11 / +0.51 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -2.23 | +0.31 / +0.19 | +0.20 / +0.31 | +0.16 | 1 |
| free_free | -1.75 | +0.53 / +0.19 | +0.28 / +0.31 | +0.24 | 2 |
| none_free | +0.00 | +0.59 / +0.19 | +0.34 / +0.31 | +0.41 | 2 |
| sharedI_free | +0.68 | +0.40 / +0.19 | +0.13 / +0.31 | +0.40 | 2 |
| shared_free | +2.42 | +0.38 / +0.19 | +0.10 / +0.31 | +0.52 | 3 |
