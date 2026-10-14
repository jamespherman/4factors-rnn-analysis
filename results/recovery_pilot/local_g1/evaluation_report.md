# Context model evaluation (results\recovery_pilot\local_g1)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +7.87 | +0.67 / +0.51 / +0.83 | +1.09 / +0.46 / +0.62 | +0.12 / +0.12 / +0.20 | +0.92 | 0.49 |
| free_free | 3 | +7.25 | +0.66 / +0.51 / +0.79 | +1.24 / +0.46 / +0.58 | +0.10 / +0.12 / -0.05 | +0.91 | 0.33 |
| none_free | 3 | +0.00 | -0.10 / +0.51 / -0.17 | +0.19 / +0.46 / +0.49 | +0.06 / +0.12 / +0.06 | +0.81 | 0.38 |
| sharedI_free | 3 | -1.28 | +0.08 / +0.51 / +0.14 | +0.50 / +0.46 / +0.35 | +0.08 / +0.12 / -0.13 | +0.85 | 0.43 |
| shared_free | 3 | +3.79 | +0.20 / +0.51 / +0.06 | +0.49 / +0.46 / +0.38 | +0.09 / +0.12 / -0.10 | +0.86 | 0.34 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.77 / +0.80 | +0.60 / +0.73 | +0.24 / +0.20 | +0.87 / +0.95 |
| free_free | +0.68 / +0.80 | +0.58 / +0.73 | +0.12 / +0.20 | +0.87 / +0.95 |
| none_free | -0.23 / +0.80 | +0.48 / +0.73 | -0.11 / +0.20 | +0.77 / +0.95 |
| sharedI_free | +0.16 / +0.80 | +0.43 / +0.73 | -0.07 / +0.20 | +0.81 / +0.95 |
| shared_free | -0.03 / +0.80 | +0.44 / +0.73 | +0.08 / +0.20 | +0.82 / +0.95 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +20.43 | +0.84 / +0.80 | +0.72 / +0.73 | +0.99 | 3 |
| free_free | +14.77 | +0.64 / +0.80 | +0.60 / +0.73 | +0.99 | 1 |
| none_free | +0.00 | +0.27 / +0.80 | +0.51 / +0.73 | +0.72 | 2 |
| sharedI_free | -0.70 | +0.48 / +0.80 | +0.42 / +0.73 | +0.91 | 1 |
| shared_free | +9.07 | +0.72 / +0.80 | +0.43 / +0.73 | +0.99 | 1 |
