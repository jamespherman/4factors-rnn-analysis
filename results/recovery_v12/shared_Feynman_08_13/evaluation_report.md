# Context model evaluation (results\recovery_v12\shared_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -3.88 | -0.03 / -0.30 / +0.52 | +0.51 / +0.24 / +0.44 | -0.10 / +0.50 / -0.15 | +0.61 | 0.47 |
| free_free | 3 | -5.17 | -0.08 / -0.30 / +0.52 | +0.58 / +0.24 / +0.55 | -0.10 / +0.50 / -0.14 | +0.68 | 0.41 |
| none_free | 3 | +0.00 | -0.01 / -0.30 / -0.11 | +0.04 / +0.24 / +0.37 | -0.07 / +0.50 / -0.13 | +0.37 | 0.31 |
| sharedI_free | 3 | +0.94 | -0.21 / -0.30 / +0.73 | +0.05 / +0.24 / +0.33 | -0.06 / +0.50 / -0.12 | +0.50 | 0.42 |
| shared_free | 3 | +0.49 | -0.22 / -0.30 / +0.74 | +0.07 / +0.24 / +0.29 | -0.05 / +0.50 / -0.08 | +0.50 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.41 / +0.41 | +0.02 / -0.09 | +0.08 / -0.17 | +0.48 / +0.91 |
| free_free | +0.35 / +0.41 | +0.02 / -0.09 | +0.07 / -0.17 | +0.54 / +0.91 |
| none_free | +0.05 / +0.41 | +0.07 / -0.09 | +0.15 / -0.17 | +0.37 / +0.91 |
| sharedI_free | +0.46 / +0.41 | -0.07 / -0.09 | +0.15 / -0.17 | +0.46 / +0.91 |
| shared_free | +0.46 / +0.41 | -0.29 / -0.09 | +0.11 / -0.17 | +0.46 / +0.91 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -5.08 | +0.37 / +0.20 | +0.31 / -0.12 | +0.69 | 1 |
| free_free | -6.39 | +0.40 / +0.20 | +0.27 / -0.12 | +0.77 | 1 |
| none_free | +0.00 | +0.40 / +0.20 | +0.21 / -0.12 | +0.48 | 1 |
| sharedI_free | +0.32 | +0.40 / +0.20 | -0.07 / -0.12 | +0.55 | 1 |
| shared_free | +0.55 | +0.38 / +0.20 | -0.14 / -0.12 | +0.52 | 1 |
