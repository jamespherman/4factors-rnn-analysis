# Context model evaluation (results\context_grid)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| free_free | 5 | -3.76 | -0.23 / +0.66 / +0.00 | +1.82 / +0.79 / -0.01 | +0.07 / -1.55 / -0.05 | +0.37 | 0.36 |
| none_free | 5 | +0.00 | +0.02 / +0.66 / +0.05 | +0.01 / +0.79 / +0.05 | -0.10 / -1.55 / +0.01 | +0.28 | 0.39 |
| none_free_lc25 | 5 | -2.08 | -0.00 / +0.66 / +0.03 | -0.01 / +0.79 / +0.07 | -0.04 / -1.55 / +0.02 | +0.29 | 0.38 |
| none_global | 5 | -11.51 | +0.04 / +0.66 / +0.02 | +0.02 / +0.79 / +0.01 | -0.14 / -1.55 / -0.02 | +0.36 | 0.41 |
| none_spatial | 5 | -14.94 | +0.02 / +0.66 / +0.04 | +0.01 / +0.79 / +0.01 | -0.02 / -1.55 / -0.08 | +0.26 | 0.40 |
| sharedE_free | 5 | +5.04 | +0.25 / +0.66 / +0.03 | +1.68 / +0.79 / -0.05 | +0.03 / -1.55 / -0.02 | +0.35 | 0.37 |
| sharedI_free | 5 | -2.85 | +0.22 / +0.66 / +0.00 | +2.25 / +0.79 / -0.05 | +0.27 / -1.55 / -0.09 | +0.38 | 0.37 |
| shared_free | 5 | +2.67 | +0.17 / +0.66 / +0.00 | +2.00 / +0.79 / -0.05 | +0.11 / -1.55 / -0.06 | +0.41 | 0.41 |
| shared_free_lc25 | 5 | -0.12 | +0.23 / +0.66 / +0.05 | +1.83 / +0.79 / -0.07 | +0.08 / -1.55 / -0.10 | +0.34 | 0.41 |
| shared_global | 5 | -12.98 | +0.02 / +0.66 / +0.02 | +2.10 / +0.79 / -0.07 | +0.07 / -1.55 / -0.08 | +0.35 | 0.36 |
| shared_spatial | 5 | -16.22 | +0.19 / +0.66 / +0.05 | +2.54 / +0.79 / -0.06 | +0.21 / -1.55 / -0.11 | +0.32 | 0.39 |
| trialvalue_free | 5 | -9.72 | +0.01 / +0.66 / +0.04 | -0.02 / +0.79 / +0.05 | +0.81 / -1.55 / -0.18 | +0.31 | 0.39 |
