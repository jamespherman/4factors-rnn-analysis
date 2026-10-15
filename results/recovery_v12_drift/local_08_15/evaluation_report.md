# Context model evaluation (results\recovery_v12_drift\local_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +1.45 | +0.29 / +0.14 / +0.26 | +3.45 / +1.95 / +0.68 | +1.46 / +2.24 / +0.49 | +0.73 | 0.41 |
| free_free | 6 | -5.57 | +0.28 / +0.14 / +0.55 | +3.44 / +1.95 / +0.72 | +1.26 / +2.24 / +0.37 | +0.68 | 0.37 |
| none_free | 6 | +0.00 | -0.07 / +0.14 / +0.03 | +0.29 / +1.95 / +0.28 | +0.21 / +2.24 / +0.19 | +0.40 | 0.44 |
| sharedI_free | 6 | +12.75 | +0.19 / +0.14 / +0.02 | +3.47 / +1.95 / +0.57 | +1.36 / +2.24 / +0.60 | +0.72 | 0.38 |
| shared_free | 6 | +11.37 | +0.34 / +0.14 / +0.06 | +3.84 / +1.95 / +0.58 | +1.29 / +2.24 / +0.48 | +0.76 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.05 / +0.24 | +0.41 / +0.37 | +0.23 / +0.48 | +0.66 / +0.88 |
| free_free | +0.16 / +0.24 | +0.46 / +0.37 | +0.19 / +0.48 | +0.63 / +0.88 |
| none_free | -0.09 / +0.24 | +0.02 / +0.37 | -0.05 / +0.48 | +0.34 / +0.88 |
| sharedI_free | -0.03 / +0.24 | +0.37 / +0.37 | +0.28 / +0.48 | +0.64 / +0.88 |
| shared_free | -0.05 / +0.24 | +0.38 / +0.37 | +0.27 / +0.48 | +0.68 / +0.88 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | -0.15 / -0.32 | +0.50 / +0.45 | +0.99 | 3 |
| free_free | -0.70 | +0.61 / +0.05 | +0.65 / +0.36 | +0.99 | 5 |
| none_free | +0.00 | +0.23 / +0.05 | +0.30 / +0.36 | +0.97 | 5 |
| sharedI_free | +3.76 | -0.40 / +0.48 | +0.28 / +0.52 | +0.86 | 4 |
| shared_free | +nan | +0.00 / -0.32 | +0.48 / +0.45 | +0.99 | 3 |
