# Context model evaluation (results\recovery_v12_drift\shared_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -21.06 | +0.91 / +1.31 / +0.36 | +1.90 / +1.51 / +0.28 | +0.02 / +0.79 / +0.15 | +0.62 | 0.42 |
| free_free | 6 | -25.45 | +1.00 / +1.31 / +0.45 | +2.36 / +1.51 / +0.31 | +0.01 / +0.79 / +0.10 | +0.65 | 0.42 |
| none_free | 6 | +0.00 | +0.10 / +1.31 / +0.16 | +0.13 / +1.51 / +0.39 | +0.02 / +0.79 / +0.21 | +0.55 | 0.47 |
| sharedI_free | 6 | -1.55 | +0.64 / +1.31 / +0.25 | +1.35 / +1.51 / +0.20 | +0.05 / +0.79 / +0.26 | +0.67 | 0.43 |
| shared_free | 6 | -0.63 | +0.63 / +1.31 / +0.22 | +1.25 / +1.51 / +0.21 | +0.04 / +0.79 / +0.15 | +0.65 | 0.45 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.26 / +0.31 | +0.16 / +0.04 | -0.04 / +0.15 | +0.36 / +0.56 |
| free_free | +0.30 / +0.31 | +0.17 / +0.04 | +0.07 / +0.15 | +0.36 / +0.56 |
| none_free | +0.13 / +0.31 | +0.18 / +0.04 | +0.09 / +0.15 | +0.35 / +0.56 |
| sharedI_free | +0.13 / +0.31 | -0.07 / +0.04 | +0.14 / +0.15 | +0.38 / +0.56 |
| shared_free | +0.03 / +0.31 | -0.06 / +0.04 | +0.06 / +0.15 | +0.38 / +0.56 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +95.32 | +0.42 / +0.07 | +0.23 / -0.02 | +0.77 | 1 |
| free_free | -40.19 | +0.13 / -0.02 | +0.02 / -0.07 | +0.12 | 4 |
| none_free | +0.00 | -0.50 / -0.02 | +0.31 / -0.07 | +0.62 | 4 |
| sharedI_free | +24.90 | -0.14 / -0.02 | +0.06 / -0.07 | +0.75 | 4 |
| shared_free | +27.90 | -0.12 / -0.02 | +0.12 / -0.07 | +0.74 | 4 |
