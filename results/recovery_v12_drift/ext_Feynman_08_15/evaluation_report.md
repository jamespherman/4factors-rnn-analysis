# Context model evaluation (results\recovery_v12_drift\ext_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | -0.06 | +0.15 / +0.15 / +0.71 | +1.02 / +0.56 / +0.64 | +0.14 / +0.26 / +0.25 | +0.76 | 0.42 |
| free_free | 6 | +0.15 | +0.18 / +0.15 / +0.70 | +1.06 / +0.56 / +0.68 | +0.19 / +0.26 / +0.27 | +0.79 | 0.32 |
| none_free | 6 | +0.00 | +0.01 / +0.15 / +0.14 | -0.01 / +0.56 / +0.12 | +0.08 / +0.26 / +0.30 | +0.70 | 0.36 |
| sharedI_free | 6 | +0.45 | +0.11 / +0.15 / +0.19 | +0.40 / +0.56 / +0.42 | +0.16 / +0.26 / +0.34 | +0.83 | 0.39 |
| shared_free | 6 | +2.10 | +0.10 / +0.15 / +0.16 | +0.36 / +0.56 / +0.37 | +0.11 / +0.26 / +0.34 | +0.83 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.25 / +0.34 | +0.40 / +0.32 | +0.06 / -0.16 | +0.73 / +0.95 |
| free_free | +0.19 / +0.34 | +0.39 / +0.32 | +0.07 / -0.16 | +0.76 / +0.95 |
| none_free | +0.02 / +0.34 | -0.10 / +0.32 | +0.12 / -0.16 | +0.65 / +0.95 |
| sharedI_free | -0.04 / +0.34 | +0.28 / +0.32 | +0.10 / -0.16 | +0.80 / +0.95 |
| shared_free | +0.03 / +0.34 | +0.26 / +0.32 | +0.09 / -0.16 | +0.80 / +0.95 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.00 | +0.23 / +0.20 | +0.65 / +0.55 | +0.99 | 5 |
| free_free | -0.80 | +0.05 / +0.20 | +0.67 / +0.55 | +0.99 | 5 |
| none_free | +0.00 | +0.32 / +0.20 | -0.26 / +0.55 | +0.99 | 5 |
| sharedI_free | -0.09 | -0.01 / +0.20 | +0.27 / +0.55 | +0.99 | 5 |
| shared_free | +0.14 | +0.09 / +0.20 | +0.20 / +0.55 | +0.98 | 5 |
