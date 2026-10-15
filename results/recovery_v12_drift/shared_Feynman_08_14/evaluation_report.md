# Context model evaluation (results\recovery_v12_drift\shared_Feynman_08_14)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -0.43 | -0.00 / +0.25 / +0.06 | +0.69 / +0.96 / +0.48 | -0.00 / -0.10 / +0.19 | +0.57 | 0.38 |
| free_free | 6 | -1.96 | +0.01 / +0.25 / +0.34 | +0.96 / +0.96 / +0.53 | -0.01 / -0.10 / +0.22 | +0.64 | 0.27 |
| none_free | 6 | +0.00 | -0.02 / +0.25 / -0.46 | +0.00 / +0.96 / -0.06 | +0.03 / -0.10 / -0.21 | +0.82 | 0.29 |
| sharedI_free | 6 | -0.17 | -0.08 / +0.25 / -0.14 | +0.45 / +0.96 / +0.51 | -0.00 / -0.10 / +0.13 | +0.60 | 0.33 |
| shared_free | 6 | -0.01 | -0.06 / +0.25 / -0.14 | +0.30 / +0.96 / +0.40 | +0.01 / -0.10 / +0.13 | +0.56 | 0.34 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.18 / +0.11 | +0.28 / +0.52 | +0.17 / +0.06 | +0.25 / +0.33 |
| free_free | -0.15 / +0.11 | +0.35 / +0.52 | +0.28 / +0.06 | +0.30 / +0.33 |
| none_free | -0.17 / +0.11 | -0.16 / +0.52 | -0.26 / +0.06 | +0.37 / +0.33 |
| sharedI_free | -0.32 / +0.11 | +0.34 / +0.52 | +0.10 / +0.06 | +0.19 / +0.33 |
| shared_free | -0.30 / +0.11 | +0.26 / +0.52 | +0.10 / +0.06 | +0.18 / +0.33 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.39 | -0.18 / +0.09 | +0.63 / +0.50 | +0.89 | 1 |
| free_free | +1.67 | -0.33 / +0.09 | +0.57 / +0.50 | +0.84 | 1 |
| none_free | +0.00 | +0.20 / +0.09 | +0.59 / +0.50 | +0.82 | 1 |
| sharedI_free | +0.83 | -0.24 / +0.09 | +0.56 / +0.50 | +0.87 | 4 |
| shared_free | +1.22 | +0.28 / +0.09 | +0.56 / +0.50 | +0.89 | 1 |
