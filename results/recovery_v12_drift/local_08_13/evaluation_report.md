# Context model evaluation (results\recovery_v12_drift\local_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +3.00 | +0.44 / +0.49 / +0.69 | +1.07 / +1.03 / +0.51 | +0.17 / +1.61 / +0.30 | +0.52 | 0.41 |
| free_free | 6 | -20.88 | +0.42 / +0.49 / +0.68 | +1.10 / +1.03 / +0.55 | +0.46 / +1.61 / +0.46 | +0.55 | 0.41 |
| none_free | 6 | +0.00 | -0.07 / +0.49 / -0.48 | +0.02 / +1.03 / +0.42 | +0.06 / +1.61 / -0.01 | +0.61 | 0.45 |
| sharedI_free | 6 | +1.28 | +0.31 / +0.49 / +0.67 | +1.13 / +1.03 / +0.68 | +0.29 / +1.61 / +0.33 | +0.62 | 0.41 |
| shared_free | 6 | -19.76 | +0.27 / +0.49 / +0.56 | +1.06 / +1.03 / +0.51 | +0.24 / +1.61 / +0.42 | +0.52 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.07 / +0.07 | +0.37 / +0.49 | -0.03 / +0.13 | +0.55 / +0.83 |
| free_free | +0.17 / +0.07 | +0.46 / +0.49 | +0.04 / +0.13 | +0.53 / +0.83 |
| none_free | +0.03 / +0.07 | +0.33 / +0.49 | -0.11 / +0.13 | +0.64 / +0.83 |
| sharedI_free | +0.01 / +0.07 | +0.55 / +0.49 | -0.01 / +0.13 | +0.65 / +0.83 |
| shared_free | +0.03 / +0.07 | +0.40 / +0.49 | +0.04 / +0.13 | +0.54 / +0.83 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +73.00 | -0.37 / -0.27 | +0.66 / +0.55 | +0.98 | 4 |
| free_free | +0.14 | +0.46 / +0.06 | +nan / +nan | +0.62 | 2 |
| none_free | +0.00 | -0.29 / +0.06 | +nan / +nan | +0.60 | 2 |
| sharedI_free | +69.17 | -0.51 / -0.27 | +0.60 / +0.55 | +0.97 | 4 |
| shared_free | +1.90 | +0.13 / +0.06 | +nan / +nan | +0.63 | 2 |
