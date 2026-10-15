# Context model evaluation (results\recovery_v12\local_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | +0.57 | +0.05 / +0.14 / +0.67 | +0.38 / -0.16 / +0.54 | +0.06 / -0.26 / -0.01 | +0.70 | 0.40 |
| free_free | 3 | +1.23 | +0.08 / +0.14 / +0.77 | +0.14 / -0.16 / +0.72 | +0.09 / -0.26 / +0.04 | +0.78 | 0.33 |
| none_free | 3 | +0.00 | -0.00 / +0.14 / +0.08 | -0.00 / -0.16 / +0.24 | +0.08 / -0.26 / -0.02 | +0.50 | 0.35 |
| sharedI_free | 3 | +0.08 | +0.04 / +0.14 / +0.08 | +0.05 / -0.16 / +0.34 | +0.09 / -0.26 / -0.04 | +0.71 | 0.31 |
| shared_free | 3 | -4.42 | +0.06 / +0.14 / +0.12 | -0.03 / -0.16 / +0.24 | +0.05 / -0.26 / -0.06 | +0.58 | 0.33 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.48 / +0.62 | +0.47 / +0.53 | -0.21 / -0.37 | +0.65 / +0.97 |
| free_free | +0.62 / +0.62 | +0.47 / +0.53 | -0.03 / -0.37 | +0.75 / +0.97 |
| none_free | +0.12 / +0.62 | +0.09 / +0.53 | -0.08 / -0.37 | +0.46 / +0.97 |
| sharedI_free | +0.17 / +0.62 | +0.46 / +0.53 | -0.10 / -0.37 | +0.70 / +0.97 |
| shared_free | +0.22 / +0.62 | +0.34 / +0.53 | -0.09 / -0.37 | +0.55 / +0.97 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +0.49 | +0.63 / +0.68 | +0.41 / +0.43 | +0.96 | 3 |
| free_free | +2.59 | +0.67 / +0.68 | +0.42 / +0.43 | +0.98 | 3 |
| none_free | +0.00 | +0.57 / +0.68 | +0.33 / +0.43 | +0.93 | 3 |
| sharedI_free | -15.34 | +0.09 / +0.54 | +nan / +nan | +0.68 | 2 |
| shared_free | -15.53 | +0.20 / +0.54 | +nan / +nan | +0.67 | 2 |
