# Context model evaluation (results\recovery_pilot\ext_g1)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +4.60 | +0.18 / +0.53 / +0.79 | +1.21 / +1.54 / +0.92 | +0.05 / +0.05 / -0.37 | +0.56 | 0.40 |
| free_free | 3 | +6.49 | +0.41 / +0.53 / +0.95 | +1.65 / +1.54 / +0.94 | +0.03 / +0.05 / -0.39 | +0.64 | 0.26 |
| none_free | 3 | +0.00 | +0.13 / +0.53 / -0.41 | -0.06 / +1.54 / +0.73 | +0.03 / +0.05 / -0.06 | +0.40 | 0.48 |
| sharedI_free | 3 | +2.07 | +0.11 / +0.53 / +0.60 | +0.52 / +1.54 / +0.79 | +0.06 / +0.05 / -0.19 | +0.66 | 0.40 |
| shared_free | 3 | +1.35 | +0.14 / +0.53 / +0.58 | +0.49 / +1.54 / +0.71 | +0.07 / +0.05 / -0.22 | +0.60 | 0.32 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.54 / +0.66 | +0.80 / +0.86 | -0.34 / +0.08 | +0.36 / +0.47 |
| free_free | +0.69 / +0.66 | +0.88 / +0.86 | -0.36 / +0.08 | +0.39 / +0.47 |
| none_free | -0.35 / +0.66 | +0.70 / +0.86 | -0.22 / +0.08 | +0.44 / +0.47 |
| sharedI_free | +0.45 / +0.66 | +0.73 / +0.86 | -0.29 / +0.08 | +0.55 / +0.47 |
| shared_free | +0.42 / +0.66 | +0.68 / +0.86 | -0.25 / +0.08 | +0.50 / +0.47 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +6.99 | +0.53 / +0.66 | +0.83 / +0.86 | +0.76 | 3 |
| free_free | +7.28 | +0.65 / +0.66 | +0.87 / +0.86 | +0.71 | 3 |
| none_free | +0.00 | -0.19 / +0.66 | +0.77 / +0.86 | +0.55 | 2 |
| sharedI_free | +4.45 | +0.62 / +0.66 | +0.74 / +0.86 | +0.84 | 3 |
| shared_free | +0.98 | +0.14 / +0.66 | +0.54 / +0.86 | +0.44 | 2 |
