# Context model evaluation (results\recovery_v12\null_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -0.63 | -0.04 / +0.19 / +0.05 | +0.75 / +0.15 / -0.08 | +0.00 / +0.15 / +0.08 | +0.60 | 0.40 |
| free_free | 3 | -5.19 | +0.09 / +0.19 / +0.39 | +0.56 / +0.15 / -0.01 | +0.00 / +0.15 / +0.07 | +0.68 | 0.29 |
| none_free | 3 | +0.00 | +0.01 / +0.19 / +0.29 | +0.00 / +0.15 / +0.28 | +0.02 / +0.15 / +0.00 | +0.28 | 0.38 |
| sharedI_free | 3 | +0.62 | +0.01 / +0.19 / -0.13 | +0.16 / +0.15 / -0.26 | +0.03 / +0.15 / +0.04 | +0.53 | 0.47 |
| shared_free | 3 | +0.25 | +0.01 / +0.19 / -0.07 | +0.16 / +0.15 / -0.23 | +0.03 / +0.15 / +0.08 | +0.53 | 0.40 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.06 / +0.21 | +0.02 / +0.01 | -0.24 / -0.06 | +0.62 / +0.98 |
| free_free | +0.10 / +0.21 | +0.00 / +0.01 | -0.17 / -0.06 | +0.70 / +0.98 |
| none_free | +0.32 / +0.21 | +0.16 / +0.01 | -0.08 / -0.06 | +0.31 / +0.98 |
| sharedI_free | -0.05 / +0.21 | -0.14 / +0.01 | -0.13 / -0.06 | +0.55 / +0.98 |
| shared_free | -0.00 / +0.21 | -0.12 / +0.01 | -0.21 / -0.06 | +0.56 / +0.98 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +12.33 | +0.45 / -0.18 | +nan / +nan | +0.68 | 2 |
| free_free | +4.98 | +0.22 / -0.18 | +nan / +nan | +0.85 | 2 |
| none_free | +0.00 | +0.51 / +0.33 | +0.29 / -0.18 | +0.19 | 3 |
| sharedI_free | +13.69 | +0.44 / -0.18 | +nan / +nan | +0.63 | 2 |
| shared_free | +11.20 | +0.46 / -0.18 | +nan / +nan | +0.58 | 2 |
