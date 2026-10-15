# Context model evaluation (results\recovery_v12_drift\ext_08_13)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +2.88 | +0.31 / +0.54 / +0.74 | +1.57 / +0.77 / +0.57 | +0.45 / +1.51 / +0.49 | +0.75 | 0.37 |
| free_free | 6 | -25.96 | +0.35 / +0.54 / +0.64 | +1.27 / +0.77 / +0.54 | +0.50 / +1.51 / +0.43 | +0.58 | 0.35 |
| none_free | 6 | +0.00 | -0.03 / +0.54 / -0.05 | +0.04 / +0.77 / +0.29 | +0.10 / +1.51 / +0.04 | +0.57 | 0.41 |
| sharedI_free | 6 | +3.25 | +0.26 / +0.54 / +0.60 | +0.93 / +0.77 / +0.36 | +0.24 / +1.51 / +0.34 | +0.71 | 0.40 |
| shared_free | 6 | -24.64 | +0.26 / +0.54 / +0.54 | +0.95 / +0.77 / +0.39 | +0.32 / +1.51 / +0.40 | +0.52 | 0.41 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.16 / +0.08 | +0.49 / +0.59 | +0.10 / +0.18 | +0.78 / +0.85 |
| free_free | +0.09 / +0.08 | +0.44 / +0.59 | +0.09 / +0.18 | +0.60 / +0.85 |
| none_free | +0.08 / +0.08 | +0.38 / +0.59 | +0.09 / +0.18 | +0.62 / +0.85 |
| sharedI_free | -0.00 / +0.08 | +0.31 / +0.59 | +0.10 / +0.18 | +0.74 / +0.85 |
| shared_free | +0.00 / +0.08 | +0.27 / +0.59 | +0.08 / +0.18 | +0.54 / +0.85 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +71.90 | -0.13 / -0.19 | +0.63 / +0.74 | +0.98 | 4 |
| free_free | +71.41 | -0.21 / -0.19 | +0.78 / +0.74 | +0.99 | 4 |
| none_free | +0.00 | +0.11 / +0.22 | +nan / +nan | +0.12 | 2 |
| sharedI_free | +3.00 | +0.04 / +0.22 | +nan / +nan | +0.71 | 2 |
| shared_free | +68.16 | -0.33 / -0.19 | +0.47 / +0.74 | +0.96 | 4 |
