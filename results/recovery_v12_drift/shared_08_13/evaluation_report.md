# Context model evaluation (results\recovery_v12_drift\shared_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -1.77 | +0.32 / +0.42 / +0.61 | +1.13 / +0.36 / +0.25 | +0.32 / +1.86 / +0.51 | +0.64 | 0.45 |
| free_free | 6 | -0.52 | +0.51 / +0.42 / +0.62 | +1.55 / +0.36 / +0.25 | +0.37 / +1.86 / +0.49 | +0.72 | 0.38 |
| none_free | 6 | +0.00 | -0.04 / +0.42 / -0.53 | +0.04 / +0.36 / +0.23 | -0.02 / +1.86 / +0.05 | +0.43 | 0.47 |
| sharedI_free | 6 | -1.72 | +0.21 / +0.42 / +0.63 | +0.65 / +0.36 / +0.10 | +0.18 / +1.86 / +0.45 | +0.48 | 0.44 |
| shared_free | 6 | -2.67 | +0.19 / +0.42 / +0.60 | +0.57 / +0.36 / +0.09 | +0.15 / +1.86 / +0.43 | +0.49 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.01 / +0.01 | +0.14 / -0.32 | +0.30 / +0.33 | +0.70 / +0.77 |
| free_free | +0.02 / +0.01 | +0.17 / -0.32 | +0.31 / +0.33 | +0.76 / +0.77 |
| none_free | -0.05 / +0.01 | +0.06 / -0.32 | +0.11 / +0.33 | +0.43 / +0.77 |
| sharedI_free | +0.03 / +0.01 | +0.05 / -0.32 | +0.18 / +0.33 | +0.46 / +0.77 |
| shared_free | -0.03 / +0.01 | +0.05 / -0.32 | +0.17 / +0.33 | +0.49 / +0.77 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -8.00 | +0.21 / -0.25 | +nan / +nan | +0.69 | 2 |
| free_free | -8.03 | +0.21 / -0.25 | +nan / +nan | +0.69 | 2 |
| none_free | +0.00 | -0.06 / -0.25 | +nan / +nan | -0.01 | 2 |
| sharedI_free | -1.72 | +0.11 / -0.25 | +nan / +nan | +0.75 | 2 |
| shared_free | -6.80 | +0.12 / -0.25 | +nan / +nan | +0.65 | 2 |
