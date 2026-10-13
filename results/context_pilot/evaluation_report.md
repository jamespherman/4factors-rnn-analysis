# Context model evaluation (results\context_pilot)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| free_free | 3 | +6.82 | +0.34 / +0.66 / +0.10 | +1.00 / +0.79 / -0.03 | -0.06 / -1.55 / -0.06 | +0.20 | 0.30 |
| none_free | 3 | +0.00 | +0.00 / +0.66 / +0.03 | -0.01 / +0.79 / +0.07 | -0.06 / -1.55 / -0.04 | +0.12 | 0.40 |
| sharedE_free | 3 | +5.04 | +0.13 / +0.66 / +0.01 | +1.21 / +0.79 / -0.01 | -0.01 / -1.55 / -0.10 | +0.22 | 0.35 |
| sharedI_free | 3 | +13.42 | +0.13 / +0.66 / +0.01 | +0.74 / +0.79 / -0.01 | -0.09 / -1.55 / -0.05 | +0.24 | 0.36 |
| shared_free | 3 | +14.43 | +0.17 / +0.66 / +0.03 | +0.99 / +0.79 / -0.03 | -0.09 / -1.55 / -0.05 | +0.25 | 0.33 |
| shared_global | 3 | +0.68 | +0.31 / +0.66 / +0.04 | +0.77 / +0.79 / -0.06 | -0.17 / -1.55 / -0.04 | +0.26 | 0.37 |
| shared_spatial | 3 | -0.66 | +0.32 / +0.66 / +0.03 | +0.71 / +0.79 / -0.05 | -0.17 / -1.55 / -0.05 | +0.28 | 0.29 |
| trialvalue_free | 3 | +1.86 | -0.03 / +0.66 / +0.06 | -0.02 / +0.79 / +0.06 | -0.57 / -1.55 / -0.16 | +0.25 | 0.41 |
