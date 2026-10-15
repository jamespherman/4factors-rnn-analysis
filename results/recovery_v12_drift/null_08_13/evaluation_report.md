# Context model evaluation (results\recovery_v12_drift\null_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +6.01 | +0.30 / -0.02 / +0.20 | +1.24 / +0.68 / +0.47 | +0.31 / +1.62 / +0.27 | +0.69 | 0.40 |
| free_free | 6 | +5.91 | +0.54 / -0.02 / +0.25 | +1.32 / +0.68 / +0.52 | +0.36 / +1.62 / +0.50 | +0.68 | 0.33 |
| none_free | 6 | +0.00 | -0.02 / -0.02 / +0.02 | -0.01 / +0.68 / +0.19 | +0.00 / +1.62 / +0.08 | +0.22 | 0.48 |
| sharedI_free | 6 | +5.33 | +0.23 / -0.02 / +0.04 | +0.68 / +0.68 / +0.30 | +0.19 / +1.62 / +0.42 | +0.41 | 0.42 |
| shared_free | 6 | +2.21 | +0.33 / -0.02 / +0.00 | +0.43 / +0.68 / +0.14 | +0.13 / +1.62 / +0.26 | +0.36 | 0.44 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.15 / +0.13 | +0.40 / -0.05 | +0.25 / +0.53 | +0.71 / +0.79 |
| free_free | -0.13 / +0.13 | +0.49 / -0.05 | +0.31 / +0.53 | +0.67 / +0.79 |
| none_free | +0.24 / +0.13 | +0.15 / -0.05 | +0.06 / +0.53 | +0.26 / +0.79 |
| sharedI_free | -0.22 / +0.13 | +0.36 / -0.05 | +0.27 / +0.53 | +0.47 / +0.79 |
| shared_free | -0.30 / +0.13 | +0.07 / -0.05 | +0.23 / +0.53 | +0.37 / +0.79 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.82 | +0.17 / +0.11 | +nan / +nan | +0.60 | 2 |
| free_free | -0.22 | +0.33 / +0.11 | +nan / +nan | +0.58 | 2 |
| none_free | +0.00 | +0.08 / +0.11 | +nan / +nan | -0.33 | 2 |
| sharedI_free | +2.60 | +0.07 / +0.11 | +nan / +nan | +0.02 | 2 |
| shared_free | +3.13 | +0.03 / +0.11 | +nan / +nan | +0.55 | 2 |
