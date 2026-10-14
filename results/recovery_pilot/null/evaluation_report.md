# Context model evaluation (results\recovery_pilot\null)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -0.62 | -0.05 / +0.01 / +0.37 | +0.94 / +0.18 / +0.61 | +0.28 / +0.03 / -0.19 | +0.89 | 0.37 |
| free_free | 3 | +0.61 | +0.11 / +0.01 / +0.34 | +1.00 / +0.18 / +0.65 | +0.11 / +0.03 / +0.09 | +0.92 | 0.25 |
| none_free | 3 | +0.00 | -0.09 / +0.01 / +0.32 | +0.29 / +0.18 / +0.46 | +0.09 / +0.03 / +0.13 | +0.88 | 0.47 |
| sharedI_free | 3 | +2.08 | -0.12 / +0.01 / +0.45 | +0.44 / +0.18 / +0.47 | +0.21 / +0.03 / +0.16 | +0.89 | 0.38 |
| shared_free | 3 | +0.96 | -0.01 / +0.01 / +0.42 | +0.33 / +0.18 / +0.71 | +0.07 / +0.03 / +0.23 | +0.90 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.27 / +0.50 | +0.77 / +0.41 | -0.35 / +0.23 | +0.85 / +0.92 |
| free_free | +0.16 / +0.50 | +0.56 / +0.41 | +0.06 / +0.23 | +0.87 / +0.92 |
| none_free | -0.02 / +0.50 | +0.35 / +0.41 | +0.11 / +0.23 | +0.85 / +0.92 |
| sharedI_free | +0.25 / +0.50 | +0.64 / +0.41 | +0.02 / +0.23 | +0.86 / +0.92 |
| shared_free | +0.12 / +0.50 | +0.52 / +0.41 | +0.24 / +0.23 | +0.87 / +0.92 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | -3.52 | +0.25 / +0.50 | +0.76 / +0.41 | +0.90 | 3 |
| free_free | -3.15 | -0.17 / +0.50 | +0.27 / +0.41 | +0.97 | 1 |
| none_free | +0.00 | +0.27 / +0.50 | -0.45 / +0.41 | +0.95 | 3 |
| sharedI_free | -0.88 | +0.34 / +0.50 | +0.34 / +0.41 | +0.93 | 3 |
| shared_free | -3.79 | +0.10 / +0.50 | +0.34 / +0.41 | +0.97 | 1 |
