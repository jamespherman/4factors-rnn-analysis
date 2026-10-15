# Context model evaluation (results\recovery_v12_drift\ext_Feynman_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -1.62 | +0.15 / +0.10 / +0.34 | +1.01 / +0.48 / +0.77 | +0.00 / +0.00 / -0.03 | +0.77 | 0.45 |
| free_free | 6 | -1.18 | +0.24 / +0.10 / +0.24 | +1.12 / +0.48 / +0.80 | +0.02 / +0.00 / +0.03 | +0.83 | 0.36 |
| none_free | 6 | +0.00 | +0.04 / +0.10 / -0.07 | +0.11 / +0.48 / +0.55 | -0.05 / +0.00 / -0.02 | +0.65 | 0.44 |
| sharedI_free | 6 | -0.65 | -0.03 / +0.10 / +0.16 | +0.24 / +0.48 / +0.30 | -0.03 / +0.00 / +0.02 | +0.58 | 0.44 |
| shared_free | 6 | +0.16 | -0.00 / +0.10 / +0.13 | +0.26 / +0.48 / +0.30 | -0.03 / +0.00 / +0.05 | +0.60 | 0.42 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.07 / -0.12 | +0.59 / +0.71 | -0.08 / -0.06 | +0.67 / +0.87 |
| free_free | -0.16 / -0.12 | +0.59 / +0.71 | -0.05 / -0.06 | +0.72 / +0.87 |
| none_free | +0.15 / -0.12 | +0.43 / +0.71 | -0.09 / -0.06 | +0.59 / +0.87 |
| sharedI_free | -0.08 / -0.12 | +0.30 / +0.71 | -0.08 / -0.06 | +0.53 / +0.87 |
| shared_free | +0.04 / -0.12 | +0.31 / +0.71 | -0.08 / -0.06 | +0.55 / +0.87 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.44 | +0.00 / -0.23 | +0.83 / +0.70 | +0.87 | 4 |
| free_free | +2.27 | -0.16 / -0.23 | +0.83 / +0.70 | +0.95 | 4 |
| none_free | +0.00 | -0.61 / -0.23 | +0.75 / +0.70 | +0.74 | 4 |
| sharedI_free | -2.12 | -0.15 / -0.23 | +0.26 / +0.70 | +0.65 | 4 |
| shared_free | -0.91 | -0.10 / -0.23 | +0.33 / +0.70 | +0.70 | 4 |
