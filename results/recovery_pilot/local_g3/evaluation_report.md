# Context model evaluation (results\recovery_pilot\local_g3)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +25.56 | +1.33 / +1.42 / +0.96 | +3.84 / +3.88 / +0.89 | +0.29 / +0.39 / +0.40 | +0.94 | 0.51 |
| free_free | 3 | +27.22 | +1.31 / +1.42 / +0.97 | +3.92 / +3.88 / +0.90 | +0.39 / +0.39 / +0.47 | +0.95 | 0.36 |
| none_free | 3 | +0.00 | +0.24 / +1.42 / +0.00 | +0.75 / +3.88 / +0.83 | +0.20 / +0.39 / +0.37 | +0.87 | 0.46 |
| sharedI_free | 3 | +10.35 | +1.15 / +1.42 / +0.80 | +2.37 / +3.88 / +0.77 | +0.16 / +0.39 / +0.39 | +0.94 | 0.49 |
| shared_free | 3 | +10.73 | +0.73 / +1.42 / +0.70 | +2.45 / +3.88 / +0.77 | +0.20 / +0.39 / +0.37 | +0.93 | 0.43 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.92 / +0.94 | +0.94 / +0.94 | +0.06 / +0.45 | +0.89 / +0.96 |
| free_free | +0.92 / +0.94 | +0.94 / +0.94 | +0.13 / +0.45 | +0.90 / +0.96 |
| none_free | +0.01 / +0.94 | +0.90 / +0.94 | +0.04 / +0.45 | +0.82 / +0.96 |
| sharedI_free | +0.75 / +0.94 | +0.88 / +0.94 | -0.00 / +0.45 | +0.90 / +0.96 |
| shared_free | +0.64 / +0.94 | +0.88 / +0.94 | -0.02 / +0.45 | +0.88 / +0.96 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +29.29 | +0.93 / +0.94 | +0.95 / +0.94 | +0.98 | 3 |
| free_free | +27.35 | +0.94 / +0.94 | +0.95 / +0.94 | +0.99 | 3 |
| none_free | +0.00 | +0.08 / +0.94 | +0.91 / +0.94 | +0.95 | 3 |
| sharedI_free | +6.44 | +0.76 / +0.94 | +0.89 / +0.94 | +0.96 | 3 |
| shared_free | +3.26 | +0.68 / +0.94 | +0.87 / +0.94 | +0.94 | 1 |
