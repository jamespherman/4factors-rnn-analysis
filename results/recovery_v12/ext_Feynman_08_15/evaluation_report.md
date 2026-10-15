# Context model evaluation (results\recovery_v12\ext_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +1.28 | -0.02 / +0.28 / +0.67 | +0.47 / +0.06 / +0.50 | +0.09 / -0.23 / -0.30 | +0.71 | 0.44 |
| free_free | 3 | -3.61 | -0.05 / +0.28 / +0.77 | +0.64 / +0.06 / +0.55 | -0.01 / -0.23 / -0.16 | +0.67 | 0.29 |
| none_free | 3 | +0.00 | +0.00 / +0.28 / +0.21 | -0.00 / +0.06 / +0.20 | +0.11 / -0.23 / -0.20 | +0.55 | 0.34 |
| sharedI_free | 3 | -0.21 | -0.03 / +0.28 / +0.10 | +0.14 / +0.06 / +0.22 | +0.10 / -0.23 / -0.19 | +0.67 | 0.41 |
| shared_free | 3 | +1.13 | -0.03 / +0.28 / +0.18 | +0.21 / +0.06 / +0.32 | +0.07 / -0.23 / -0.40 | +0.70 | 0.23 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.43 / +0.59 | +0.43 / +0.42 | -0.18 / +0.10 | +0.69 / +0.98 |
| free_free | +0.49 / +0.59 | +0.52 / +0.42 | +0.04 / +0.10 | +0.65 / +0.98 |
| none_free | +0.24 / +0.59 | +0.27 / +0.42 | -0.17 / +0.10 | +0.52 / +0.98 |
| sharedI_free | +0.22 / +0.59 | +0.27 / +0.42 | -0.17 / +0.10 | +0.64 / +0.98 |
| shared_free | +0.11 / +0.59 | +0.24 / +0.42 | -0.21 / +0.10 | +0.67 / +0.98 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +2.28 | +0.43 / +0.50 | +0.37 / +0.42 | +0.98 | 3 |
| free_free | -12.05 | +0.39 / +0.50 | +nan / +nan | +0.69 | 2 |
| none_free | +0.00 | +0.54 / +0.50 | +0.28 / +0.42 | +0.99 | 3 |
| sharedI_free | -0.53 | +0.31 / +0.50 | +0.28 / +0.42 | +0.99 | 3 |
| shared_free | -10.69 | -0.10 / +0.50 | +nan / +nan | +0.54 | 2 |
