# Context model evaluation (results\recovery_v12\ext_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +0.66 | +0.05 / -0.00 / +0.76 | +0.51 / +0.74 / +0.78 | +0.02 / +0.06 / +0.42 | +0.88 | 0.45 |
| free_free | 3 | +0.47 | -0.04 / -0.00 / +0.78 | +0.75 / +0.74 / +0.83 | +0.01 / +0.06 / +0.37 | +0.89 | 0.33 |
| none_free | 3 | +0.00 | -0.01 / -0.00 / -0.78 | +0.02 / +0.74 / +0.52 | +0.01 / +0.06 / +0.35 | +0.82 | 0.37 |
| sharedI_free | 3 | +0.10 | +0.12 / -0.00 / +0.37 | +0.37 / +0.74 / +0.66 | +0.01 / +0.06 / +0.43 | +0.84 | 0.30 |
| shared_free | 3 | +0.08 | +0.10 / -0.00 / +0.49 | +0.38 / +0.74 / +0.67 | +0.01 / +0.06 / +0.39 | +0.84 | 0.31 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.36 / +0.47 | +0.44 / +0.24 | +0.23 / +0.23 | -0.01 / -0.08 |
| free_free | +0.41 / +0.47 | +0.47 / +0.24 | +0.18 / +0.23 | -0.01 / -0.08 |
| none_free | -0.38 / +0.47 | +0.45 / +0.24 | +0.07 / +0.23 | -0.05 / -0.08 |
| sharedI_free | +0.31 / +0.47 | +0.45 / +0.24 | +0.11 / +0.23 | -0.04 / -0.08 |
| shared_free | +0.35 / +0.47 | +0.46 / +0.24 | +0.11 / +0.23 | -0.03 / -0.08 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.28 | -0.10 / +0.10 | +0.39 / +0.48 | +0.88 | 3 |
| free_free | +0.22 | -0.08 / +0.10 | +0.54 / +0.48 | +0.88 | 3 |
| none_free | +0.00 | +0.19 / +0.10 | +0.29 / +0.48 | +0.83 | 3 |
| sharedI_free | +0.05 | -0.13 / +0.10 | +0.52 / +0.48 | +0.80 | 3 |
| shared_free | +0.05 | -0.16 / +0.10 | +0.49 / +0.48 | +0.78 | 3 |
