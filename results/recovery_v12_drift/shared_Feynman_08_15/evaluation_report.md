# Context model evaluation (results\recovery_v12_drift\shared_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +0.37 | +0.20 / -0.11 / +0.53 | +1.26 / +0.37 / +0.54 | +0.19 / +0.48 / +0.58 | +0.85 | 0.47 |
| free_free | 6 | -2.23 | +0.07 / -0.11 / +0.57 | +1.34 / +0.37 / +0.60 | +0.22 / +0.48 / +0.62 | +0.84 | 0.33 |
| none_free | 6 | +0.00 | +0.00 / -0.11 / +0.26 | +0.00 / +0.37 / +0.28 | +0.03 / +0.48 / +0.39 | +0.56 | 0.37 |
| sharedI_free | 6 | +0.21 | -0.02 / -0.11 / +0.14 | +0.91 / +0.37 / +0.50 | +0.14 / +0.48 / +0.54 | +0.82 | 0.40 |
| shared_free | 6 | +1.62 | -0.04 / -0.11 / +0.13 | +1.05 / +0.37 / +0.49 | +0.13 / +0.48 / +0.48 | +0.84 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.11 / +0.27 | +0.17 / +0.21 | +0.27 / +0.20 | +0.79 / +0.92 |
| free_free | +0.12 / +0.27 | +0.18 / +0.21 | +0.25 / +0.20 | +0.77 / +0.92 |
| none_free | +0.19 / +0.27 | +0.17 / +0.21 | +0.03 / +0.20 | +0.50 / +0.92 |
| sharedI_free | -0.00 / +0.27 | +0.16 / +0.21 | +0.28 / +0.20 | +0.77 / +0.92 |
| shared_free | -0.01 / +0.27 | +0.13 / +0.21 | +0.16 / +0.20 | +0.80 / +0.92 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -3.35 | +0.39 / +0.59 | +0.49 / +0.14 | +0.99 | 5 |
| free_free | -3.25 | +0.40 / +0.59 | +0.53 / +0.14 | +0.99 | 5 |
| none_free | +0.00 | +0.36 / +0.59 | -0.31 / +0.14 | +0.97 | 5 |
| sharedI_free | -3.65 | -0.37 / +0.59 | +0.56 / +0.14 | +0.99 | 5 |
| shared_free | -1.83 | -0.39 / +0.59 | +0.55 / +0.14 | +0.99 | 5 |
