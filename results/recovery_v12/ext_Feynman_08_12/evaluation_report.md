# Context model evaluation (results\recovery_v12\ext_Feynman_08_12)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +17.41 | +0.39 / +0.38 / +0.63 | +0.83 / +1.29 / +0.71 | -0.03 / +0.01 / +0.44 | +0.83 | 0.20 |
| free_free | 3 | +18.89 | +0.59 / +0.38 / +0.76 | +1.03 / +1.29 / +0.87 | -0.01 / +0.01 / +0.27 | +0.92 | 0.13 |
| none_free | 3 | +0.00 | +0.01 / +0.38 / -0.45 | +0.06 / +1.29 / +0.00 | -0.01 / +0.01 / +0.33 | +0.30 | 0.23 |
| sharedI_free | 3 | +15.71 | +0.20 / +0.38 / +0.26 | +0.71 / +1.29 / +0.61 | -0.02 / +0.01 / +0.30 | +0.80 | 0.13 |
| shared_free | 3 | +10.33 | +0.08 / +0.38 / +0.06 | +1.02 / +1.29 / +0.54 | +0.02 / +0.01 / +0.27 | +0.89 | 0.23 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.13 / +0.23 | +0.31 / +0.29 | +0.17 / -0.10 | +0.49 / +0.54 |
| free_free | +0.34 / +0.23 | +0.24 / +0.29 | +0.42 / -0.10 | +0.53 / +0.54 |
| none_free | +0.09 / +0.23 | +0.05 / +0.29 | +0.34 / -0.10 | +0.08 / +0.54 |
| sharedI_free | +0.38 / +0.23 | +0.17 / +0.29 | +0.43 / -0.10 | +0.46 / +0.54 |
| shared_free | +0.11 / +0.23 | +0.11 / +0.29 | +0.40 / -0.10 | +0.51 / +0.54 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +48.74 | -0.32 / -0.05 | +0.14 / +0.27 | +0.94 | 1 |
| free_free | +49.49 | +0.30 / -0.05 | +0.38 / +0.27 | +0.95 | 1 |
| none_free | +0.00 | +0.52 / -0.05 | -0.16 / +0.27 | -0.78 | 1 |
| sharedI_free | +45.15 | +0.18 / -0.05 | -0.23 / +0.27 | +0.83 | 1 |
| shared_free | +49.20 | +0.11 / -0.05 | -0.54 / +0.27 | +0.95 | 1 |
