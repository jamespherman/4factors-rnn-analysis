# Context model evaluation (results\recovery_v12_drift\ext_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -2.38 | +0.58 / +0.66 / +0.52 | +0.98 / +0.69 / +0.71 | +0.02 / +0.00 / +0.38 | +0.68 | 0.46 |
| free_free | 6 | -4.40 | +0.51 / +0.66 / +0.58 | +1.19 / +0.69 / +0.72 | +0.03 / +0.00 / +0.26 | +0.70 | 0.40 |
| none_free | 6 | +0.00 | +0.04 / +0.66 / +0.25 | +0.04 / +0.69 / +0.26 | -0.02 / +0.00 / +0.13 | +0.53 | 0.52 |
| sharedI_free | 6 | +0.49 | +0.20 / +0.66 / +0.37 | +0.42 / +0.69 / +0.39 | +0.02 / +0.00 / +0.24 | +0.61 | 0.43 |
| shared_free | 6 | +2.33 | +0.20 / +0.66 / +0.30 | +0.87 / +0.69 / +0.30 | -0.01 / +0.00 / +0.26 | +0.66 | 0.47 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.14 / +0.10 | +0.45 / +0.42 | +0.23 / +0.31 | +0.49 / +0.68 |
| free_free | +0.16 / +0.10 | +0.45 / +0.42 | +0.27 / +0.31 | +0.48 / +0.68 |
| none_free | -0.10 / +0.10 | +0.14 / +0.42 | +0.02 / +0.31 | +0.41 / +0.68 |
| sharedI_free | +0.11 / +0.10 | +0.30 / +0.42 | +0.13 / +0.31 | +0.50 / +0.68 |
| shared_free | +0.07 / +0.10 | +0.13 / +0.42 | +0.07 / +0.31 | +0.49 / +0.68 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +10.16 | -0.34 / -0.36 | +0.35 / +0.39 | +0.81 | 4 |
| free_free | +7.58 | -0.28 / -0.36 | +0.39 / +0.39 | +0.76 | 4 |
| none_free | +0.00 | -0.58 / -0.36 | -0.13 / +0.39 | +0.78 | 4 |
| sharedI_free | +10.27 | +0.00 / -0.36 | +0.33 / +0.39 | +0.90 | 4 |
| shared_free | +13.94 | -0.20 / -0.36 | +0.16 / +0.39 | +0.81 | 4 |
