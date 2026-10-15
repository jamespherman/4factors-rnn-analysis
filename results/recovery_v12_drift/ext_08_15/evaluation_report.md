# Context model evaluation (results\recovery_v12_drift\ext_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 6 | +0.89 | +0.44 / -0.10 / +0.39 | +2.94 / +1.33 / +0.74 | +1.29 / +1.72 / +0.77 | +0.84 | 0.41 |
| free_free | 6 | +38.09 | +0.39 / -0.10 / +0.67 | +2.71 / +1.33 / +0.79 | +1.52 / +1.72 / +0.83 | +0.91 | 0.35 |
| none_free | 6 | +0.00 | -0.05 / -0.10 / +0.06 | +0.22 / +1.33 / +0.22 | +0.18 / +1.72 / +0.10 | +0.54 | 0.40 |
| sharedI_free | 6 | +6.68 | +0.50 / -0.10 / +0.04 | +3.51 / +1.33 / +0.66 | +1.07 / +1.72 / +0.66 | +0.83 | 0.37 |
| shared_free | 6 | -15.17 | +0.50 / -0.10 / +0.08 | +3.34 / +1.33 / +0.54 | +0.77 / +1.72 / +0.42 | +0.74 | 0.39 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | +0.11 / +0.30 | +0.45 / +0.41 | +0.38 / +0.48 | +0.78 / +0.91 |
| free_free | +0.24 / +0.30 | +0.54 / +0.41 | +0.41 / +0.48 | +0.85 / +0.91 |
| none_free | -0.04 / +0.30 | +0.04 / +0.41 | -0.12 / +0.48 | +0.50 / +0.91 |
| sharedI_free | +0.11 / +0.30 | +0.38 / +0.41 | +0.26 / +0.48 | +0.77 / +0.91 |
| shared_free | +0.26 / +0.30 | +0.36 / +0.41 | +0.16 / +0.48 | +0.71 / +0.91 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +nan | -0.10 / -0.16 | +0.54 / +0.41 | +0.99 | 3 |
| free_free | +37.42 | +0.55 / +0.12 | +0.73 / +0.43 | +1.00 | 5 |
| none_free | +0.00 | -0.48 / +0.47 | +nan / +nan | +0.98 | 6 |
| sharedI_free | +21.79 | +0.35 / +0.47 | +nan / +nan | +0.99 | 6 |
| shared_free | +nan | -0.05 / -0.16 | +0.57 / +0.41 | +0.99 | 3 |
