# Context model evaluation (results\recovery_v12\shared_Feynman_08_15)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| freeI_free | 3 | -1.16 | -0.03 / -0.13 / +0.19 | +0.55 / +0.04 / +0.29 | +0.01 / -0.09 / -0.13 | +0.61 | 0.42 |
| free_free | 3 | -5.80 | -0.03 / -0.13 / +0.30 | +0.38 / +0.04 / +0.28 | -0.02 / -0.09 / -0.12 | +0.57 | 0.24 |
| none_free | 3 | +0.00 | +0.00 / -0.13 / +0.27 | +0.01 / +0.04 / +0.32 | -0.03 / -0.09 / -0.09 | +0.46 | 0.34 |
| sharedI_free | 3 | +1.73 | -0.04 / -0.13 / -0.12 | +0.18 / +0.04 / +0.07 | +0.02 / -0.09 / -0.10 | +0.57 | 0.46 |
| shared_free | 3 | +0.13 | -0.07 / -0.13 / -0.01 | +0.18 / +0.04 / +0.13 | +0.01 / -0.09 / -0.11 | +0.58 | 0.44 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| freeI_free | -0.05 / +0.39 | +0.27 / +0.32 | -0.27 / -0.12 | +0.63 / +0.96 |
| free_free | +0.10 / +0.39 | +0.26 / +0.32 | -0.26 / -0.12 | +0.59 / +0.96 |
| none_free | +0.38 / +0.39 | +0.13 / +0.32 | -0.29 / -0.12 | +0.49 / +0.96 |
| sharedI_free | +0.12 / +0.39 | +0.19 / +0.32 | -0.34 / -0.12 | +0.59 / +0.96 |
| shared_free | +0.24 / +0.39 | +0.22 / +0.32 | -0.26 / -0.12 | +0.60 / +0.96 |

Best restart per variant and session (highest selection-set log-likelihood); held-out r / ceiling as above:

| Variant | Delta LL x1000 | Reward pre r | Block pre r | Spatial r (all trials) | Chosen seeds |
|---|---|---|---|---|---|
| freeI_free | +1.90 | +0.09 / +0.00 | +nan / +nan | +0.74 | 2 |
| free_free | +0.85 | -0.04 / +0.00 | +nan / +nan | +0.87 | 2 |
| none_free | +0.00 | +0.17 / +0.00 | +nan / +nan | +0.52 | 2 |
| sharedI_free | +1.44 | +0.08 / +0.00 | +nan / +nan | +0.59 | 2 |
| shared_free | +1.26 | +0.07 / +0.00 | +nan / +nan | +0.60 | 2 |
