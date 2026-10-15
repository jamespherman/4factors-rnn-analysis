# Context model evaluation (results\recovery_v12_drift\null_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -23.83 | +0.24 / +0.34 / -0.06 | +1.81 / +1.52 / +0.21 | +0.02 / +0.61 / +0.17 | +0.53 | 0.44 |
| free_free | 6 | -23.01 | +0.15 / +0.34 / -0.09 | +1.91 / +1.52 / +0.24 | +0.03 / +0.61 / +0.19 | +0.57 | 0.37 |
| none_free | 6 | +0.00 | +0.05 / +0.34 / +0.02 | +0.13 / +1.52 / +0.19 | -0.02 / +0.61 / +0.10 | +0.48 | 0.52 |
| sharedI_free | 6 | +1.23 | +0.06 / +0.34 / -0.03 | +0.70 / +1.52 / -0.01 | +0.03 / +0.61 / +0.17 | +0.54 | 0.49 |
| shared_free | 6 | -10.82 | +0.14 / +0.34 / -0.03 | +0.67 / +1.52 / +0.06 | +0.02 / +0.61 / +0.09 | +0.45 | 0.44 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.14 / -0.08 | +0.12 / -0.02 | +0.05 / +0.16 | +0.29 / +0.54 |
| free_free | +0.09 / -0.08 | +0.17 / -0.02 | +0.03 / +0.16 | +0.29 / +0.54 |
| none_free | +0.12 / -0.08 | +0.05 / -0.02 | +0.01 / +0.16 | +0.32 / +0.54 |
| sharedI_free | -0.07 / -0.08 | -0.09 / -0.02 | +0.10 / +0.16 | +0.33 / +0.54 |
| shared_free | +0.03 / -0.08 | +0.07 / -0.02 | -0.02 / +0.16 | +0.23 / +0.54 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -62.95 | +0.00 / -0.11 | +0.27 / -0.04 | +0.18 | 4 |
| free_free | -51.58 | +0.07 / -0.11 | +0.33 / -0.04 | +0.33 | 4 |
| none_free | +0.00 | +0.34 / -0.11 | +0.19 / -0.04 | +0.71 | 4 |
| sharedI_free | +10.46 | +0.09 / -0.11 | +0.04 / -0.04 | +0.76 | 4 |
| shared_free | -4.00 | +0.01 / -0.11 | +0.05 / -0.04 | +0.75 | 4 |
