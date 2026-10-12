# Optimization Findings: Training and Hyperparameters

## Executive Summary

This document focuses specifically on training/optimization parameters and their effects on model performance. The optimal configuration uses:
- **Learning rate:** 0.001 (Adam default)
- **Scheduler:** ReduceLROnPlateau (patience=50, factor=0.5)
- **Epochs:** 1000-2000 (more is better, but diminishing returns)
- **Early stopping patience:** 200-300
- **Gradient clipping:** 1.0

---

## 1. Learning Rate

### Values Tested
| Learning Rate | Experiment | Val Corr | Relative |
|---------------|------------|----------|----------|
| 0.001 | Default (all phases) | 0.3762-0.4121 | 100% |
| 0.0005 | embed_56_lower_lr | 0.3694 | 98% |

### Findings
- **0.001 (Adam default) is optimal**
- Lower learning rate (0.0005) showed slight degradation (-2%)
- No experiments tested higher learning rates (risky with RNNs)

### Recommendation
```python
learning_rate = 0.001
```

---

## 2. Learning Rate Schedulers

### Schedulers Tested
| Scheduler | Warmup | Experiment | Val Corr | Relative |
|-----------|--------|------------|----------|----------|
| Plateau | N/A | All best configs | 0.4121 | 100% |
| Cosine | 50 epochs | cosine_warmup_50 | 0.3051 | 74% |
| Cosine | 0 epochs | cosine_no_warmup | 0.2801 | 68% |

### Findings
- **ReduceLROnPlateau is consistently best**
- Cosine annealing without warmup hurt significantly (-32%)
- Cosine with warmup still underperformed (-26%)

### Optimal Plateau Configuration
```python
scheduler = 'plateau'
lr_scheduler_patience = 50      # Epochs to wait before reducing
lr_scheduler_factor = 0.5       # Multiply LR by this when reducing
min_lr = 1e-5                   # Minimum learning rate floor
```

---

## 3. Number of Epochs

### Training Duration Analysis
| Max Epochs | Best Epoch | Val Corr | Status |
|------------|------------|----------|--------|
| 500 | varies | 0.3762-0.4021 | Converged for most configs |
| 1000 | ~1000 | 0.4121 | Still improving |
| 2000 | 959 | 0.4070 | Optimal for final model |

### Findings
- **500 epochs:** Sufficient for most architecture comparisons
- **1000 epochs:** Extended training improved best config by +2.5%
- **2000 epochs:** Final model's best at epoch 959, suggesting 1000 optimal
- Diminishing returns after ~1000 epochs

### Recommendation
```python
# For quick experiments
max_epochs = 500
patience = 100

# For final/production models
max_epochs = 2000
patience = 300
```

---

## 4. Early Stopping

### Patience Values Tested
| Patience | Context | Outcome |
|----------|---------|---------|
| 100 | Phase 2-4 experiments | Good for comparisons |
| 150 | Early tests | Adequate |
| 200 | Phase 5 | Better for extended training |
| 300 | Final model | Optimal for production |

### Findings
- Early stopping is **essential** - prevents overfitting
- Patience should scale with max_epochs
- Final model used patience=300 with max_epochs=2000
- Best epoch (959) shows early stopping triggered appropriately

### Recommendation
```python
patience = int(0.15 * max_epochs)  # ~15% of max epochs
```

---

## 5. Gradient Clipping

### Configuration
```python
gradient_clip = 1.0  # Max gradient norm
```

### Findings
- All experiments used gradient_clip=1.0
- No systematic comparison of different values
- 1.0 appears sufficient for stable training
- No reported gradient explosion issues

### Status
Untested variations. Current value (1.0) works well.

---

## 6. Batch Size

### Configuration
All experiments used **full-batch training** (all trials in one batch):
- Newton_08_15: 835 train trials, 208 val trials
- Newton_08_14: 430 train trials, 106 val trials
- Newton_08_13: 330 train trials, 82 val trials

### Findings
- Full-batch required for L_trial (trial-matching needs all trials)
- No mini-batch experiments conducted
- Memory not a bottleneck with current network sizes

### Status
Full-batch is mandatory for trial-matching loss.

---

## 7. Optimizer

### Configuration
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### AdamW Comparison
| Optimizer | Experiment | Val Corr | Effect |
|-----------|------------|----------|--------|
| Adam | Default | 0.3762 | baseline |
| AdamW | embed_56_adamw | 0.3748 | -0.4% |
| AdamW | attention_adamw | 0.3878 | +0.9% from attention baseline |

### Findings
- **Adam and AdamW perform similarly**
- AdamW (weight decay) provides marginal regularization benefit
- AdamW combined with learnable h0 slightly hurt (0.3987 vs 0.4021)

### Recommendation
Adam is fine. AdamW optional for mild regularization.

---

## 8. Regularization

### L2 Regularization
```python
lambda_reg = 0.0001  # Default
```

### Activity Regularization
| Lambda | Val Corr | Effect |
|--------|----------|--------|
| 0.0 | 0.2901 | baseline |
| 0.001 | 0.3171 | +9.3% |
| 0.01 | 0.2828 | -2.5% |

### Findings
- **Mild L2 regularization (1e-4) is standard**
- Activity regularization at 0.001 helped slightly
- Activity regularization at 0.01 was too strong
- Best configs don't use explicit activity regularization

### Recommendation
```python
lambda_reg = 0.0001  # L2 on weights
# Activity regularization: optional, use 0.001 if needed
```

---

## 9. L_trial Weighting (Gradient Balancing)

### Critical Finding
**Gradient balancing is essential for L_trial optimization.**

### Comparison
| Method | Val Corr |
|--------|----------|
| No L_trial | 0.3299 |
| Fixed lambda=0.1 | 0.3185 |
| Fixed lambda=0.3 | 0.1877 |
| Gradient balancing (scale=1.0) | 0.3434 |
| Gradient balancing (scale=0.5) | **0.3614** |

### Optimal Configuration
```python
use_grad_balancing = True
ltrial_scale = 0.5              # Reduce L_trial contribution
sinkhorn_epsilon = 0.1          # Tight assignments
```

### Warmup Experiments (All Hurt)
| Warmup Epochs | Val Corr |
|---------------|----------|
| 0 (no warmup) | 0.3434 |
| 50 | 0.2949 |
| 100 | 0.3168 |

**Conclusion:** No warmup needed. Include L_trial from epoch 0.

---

## 10. Training Stability

### Metrics
| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Convergence | 80-100% of experiments | Very stable |
| Failed runs | 2 of ~100 | tau<dt bug (fixed) |
| Gradient explosion | None reported | Clipping effective |

### Failure Modes
1. **tau < dt:** Caused alpha > 1, breaking dynamics (fixed by TAU_MIN >= dt)
2. **Very soft Sinkhorn:** Led to poor trial matching
3. **High fixed lambda:** Gradient conflicts prevented convergence

---

## Summary: Optimal Training Configuration

```python
# Optimizer
optimizer = 'adam'
learning_rate = 0.001
gradient_clip = 1.0

# Scheduler
scheduler = 'plateau'
lr_scheduler_patience = 50
lr_scheduler_factor = 0.5
min_lr = 1e-5

# Training duration
max_epochs = 2000      # For production
patience = 300         # Early stopping

# Loss weighting
use_grad_balancing = True
ltrial_scale = 0.5
sinkhorn_epsilon = 0.1

# Regularization
lambda_reg = 0.0001
```

---

## Untested Optimizations

The following were not systematically explored:

1. **Higher learning rates (0.01, 0.003)**
2. **Different optimizers (SGD+momentum, RMSprop)**
3. **Learning rate warmup**
4. **Gradient clipping values (0.5, 2.0, 5.0)**
5. **Mini-batch training** (incompatible with current L_trial)
6. **Curriculum learning** (tested briefly, config files suggest no benefit)
7. **Label smoothing or other regularization**
