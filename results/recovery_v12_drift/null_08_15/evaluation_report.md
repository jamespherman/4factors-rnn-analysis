# Context model evaluation (results\recovery_v12_drift\null_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +1.87 | -0.13 / -0.31 / +0.18 | +2.50 / +1.34 / +0.37 | +0.40 / +3.70 / +0.17 | +0.47 | 0.43 |
| free_free | 6 | +7.81 | -0.54 / -0.31 / +0.39 | +3.08 / +1.34 / +0.56 | +1.63 / +3.70 / +0.63 | +0.71 | 0.40 |
| none_free | 6 | +0.00 | -0.07 / -0.31 / -0.01 | +0.20 / +1.34 / +0.24 | +0.35 / +3.70 / +0.11 | +0.40 | 0.43 |
| sharedI_free | 6 | +2.72 | -0.21 / -0.31 / +0.03 | +2.67 / +1.34 / +0.37 | +0.42 / +3.70 / +0.33 | +0.48 | 0.36 |
| shared_free | 6 | +7.63 | -0.52 / -0.31 / +0.20 | +3.49 / +1.34 / +0.35 | +1.06 / +3.70 / +0.30 | +0.53 | 0.40 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.09 / +0.10 | +0.17 / +0.11 | +0.05 / +0.51 | +0.43 / +0.83 |
| free_free | -0.02 / +0.10 | +0.20 / +0.11 | +0.28 / +0.51 | +0.62 / +0.83 |
| none_free | -0.15 / +0.10 | +0.07 / +0.11 | +0.06 / +0.51 | +0.37 / +0.83 |
| sharedI_free | -0.16 / +0.10 | +0.10 / +0.11 | +0.22 / +0.51 | +0.43 / +0.83 |
| shared_free | +0.00 / +0.10 | +0.11 / +0.11 | +0.01 / +0.51 | +0.46 / +0.83 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +12.31 | -0.10 / +0.17 | +0.35 / +0.18 | +0.84 | 4 |
| free_free | +16.79 | +0.31 / +0.17 | +0.39 / +0.18 | +0.88 | 4 |
| none_free | +0.00 | -0.49 / -0.07 | +nan / +nan | +0.98 | 6 |
| sharedI_free | +15.95 | -0.53 / +0.17 | +0.45 / +0.18 | +0.89 | 4 |
| shared_free | +17.72 | +0.23 / +0.15 | -0.15 / -0.27 | +0.55 | 2 |
