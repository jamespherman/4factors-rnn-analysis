# E-I RNN Improvement Plan - Phase 2c
## January 23, 2026

This document provides complete context for continuing optimization work on the E-I RNN model for SC neural data fitting. It is designed to be self-contained for a fresh context window.

---

## Project Overview

### Goal
Train an Excitatory-Inhibitory RNN to fit neural recordings from Superior Colliculus (SC), matching both:
1. **Trial-averaged activity (PSTH)** - Mean firing rates across trials
2. **Single-trial variability** - Trial-to-trial differences in population trajectories

### Performance Metric
- **Primary:** Validation PSTH correlation (Pearson r, averaged across neurons)

### Current Status
| Milestone | Val Correlation | Date |
|-----------|-----------------|------|
| Target (reported best) | 0.3614 | Prior work |
| **Current best** | **0.3449** | Jan 23, 2026 |
| Phase 1 baseline | 0.2901 | Jan 23, 2026 |
| Gap to target | 4.6% | - |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | EIRNN class with Dale's law, learnable alpha, low-rank |
| `src/losses.py` | Loss functions (L_neuron, L_trial with Poisson option) |
| `src/data_loader.py` | Data loading and preprocessing |
| `scripts/experiment_phase2.py` | Phase 2 experiment script |
| `data/rnn_export_Newton_08_15_2025_SC.mat` | Neural data (1043 trials, 54 neurons, 219 time bins) |
| `results/phase2/summary.json` | Phase 2 results |
| `specs/improvement_experiment_results_01232026.md` | Detailed Phase 2 analysis |

---

## Phase 2 Results Summary

### What Worked (12 improvements)

| Rank | Improvement | Val Corr | vs Baseline | Key Insight |
|------|-------------|----------|-------------|-------------|
| 1 | learnable_alpha_neuron | 0.3449 | +18.9% | E: 0.54, I: 0.72 |
| 2 | learnable_alpha_population | 0.3443 | +18.7% | E: 0.58, I: 0.73 |
| 3 | poisson_ltrial | 0.3369 | +16.1% | Poisson Bregman OT |
| 4 | low_rank_50 | 0.3334 | +14.9% | rank=50 best |
| 5 | low_rank_20 | 0.3227 | +11.2% | |
| 6 | activity_reg_0.001 | 0.3171 | +9.3% | Light regularization |

### What Failed

| Approach | Result | Reason |
|----------|--------|--------|
| learnable_tau | 0.02 | α > 1 when τ < dt=25ms |
| cosine_no_warmup | 0.2801 | LR drops too fast |
| activity_reg_0.01 | 0.2828 | Too strong |

### Untested Combinations

The two combinations tested did NOT stack benefits:
- poisson_ltrial + alpha_pop → 0.3171 (< either alone)
- low_rank_20 + alpha_pop → 0.3145 (< either alone)

**However, many promising combinations were NOT tested:**
- learnable_alpha_neuron + poisson_ltrial
- learnable_alpha_neuron + low_rank_50
- Triple combinations
- Combinations with optimal hyperparameters

---

## Phase 2c Plan: Systematic Combination Search + Input Embedding

### Strategy

1. **Systematic combination testing** of the top individual improvements
2. **Input embedding expansion** based on Cover's theorem
3. **Fix learnable tau** by constraining τ ≥ dt
4. **Hyperparameter refinement** using insights from learned values

---

## Improvement 1: Systematic Combination Testing

### Rationale
12 out of 17 experiments improved over baseline. The key question is whether improvements can stack. The limited combinations tested (2 out of many possible) showed no stacking, but this may be due to:
- Suboptimal pairing (poisson_ltrial may conflict with alpha learning)
- Missing synergistic combinations
- Need for hyperparameter tuning when combining

### Priority Combinations to Test

**Tier 1 (Most Promising):**
```python
combinations_tier1 = [
    # Top 2 performers that address different aspects
    {
        'name': 'alpha_neuron_poisson',
        'learnable_alpha': 'neuron',
        'use_poisson_ltrial': True,
        'alpha_init': 0.5,  # Or 0.54 for E, 0.72 for I
    },
    # Best alpha + best regularizer
    {
        'name': 'alpha_neuron_lowrank50',
        'learnable_alpha': 'neuron',
        'low_rank': 50,
    },
    # Poisson + low-rank (no alpha learning)
    {
        'name': 'poisson_lowrank50',
        'use_poisson_ltrial': True,
        'low_rank': 50,
    },
]
```

**Tier 2 (Triple Combinations):**
```python
combinations_tier2 = [
    # All three top improvements
    {
        'name': 'alpha_neuron_poisson_lowrank50',
        'learnable_alpha': 'neuron',
        'use_poisson_ltrial': True,
        'low_rank': 50,
    },
    # With light activity reg
    {
        'name': 'alpha_neuron_poisson_actreg',
        'learnable_alpha': 'neuron',
        'use_poisson_ltrial': True,
        'use_activity_reg': True,
        'lambda_activity': 0.001,
    },
]
```

**Tier 3 (Hyperparameter Variants):**
```python
combinations_tier3 = [
    # Initialize alpha closer to learned optimal
    {
        'name': 'alpha_neuron_optimal_init',
        'learnable_alpha': 'neuron',
        'alpha_e_init': 0.54,  # From Phase 2 learned value
        'alpha_i_init': 0.72,  # From Phase 2 learned value
    },
    # Lower learning rate for alpha params
    {
        'name': 'alpha_neuron_low_lr',
        'learnable_alpha': 'neuron',
        'alpha_lr_scale': 0.1,  # 10x lower LR for alpha
    },
]
```

### Implementation

Modify `scripts/experiment_phase2.py` or create new `scripts/experiment_phase2c.py`:

```python
# Add to model creation
model = create_model_from_data(
    ...,
    learnable_alpha='neuron',
    alpha_e_init=config.get('alpha_e_init', 0.5),
    alpha_i_init=config.get('alpha_i_init', 0.75),
    low_rank=config.get('low_rank', None),
)

# Separate learning rates for alpha parameters
if config.get('alpha_lr_scale'):
    alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
    other_params = [p for n, p in model.named_parameters() if 'alpha' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': alpha_params, 'lr': lr * config['alpha_lr_scale']}
    ])
```

---

## Improvement 2: Input Embedding Expansion (Cover's Theorem)

### Rationale

**Cover's Theorem:** A complex pattern classification problem is more likely to be linearly separable when cast into a higher-dimensional space via a nonlinear transformation.

The current model receives 14-dimensional input signals directly. Expanding the input dimensionality through a learnable or fixed nonlinear embedding may help the RNN find better representations of the task variables.

### Proposed Approaches

**Approach A: Learnable Linear Expansion**
```python
class InputEmbedding(nn.Module):
    def __init__(self, n_inputs, embed_dim, nonlinearity='relu'):
        super().__init__()
        self.linear = nn.Linear(n_inputs, embed_dim)
        self.nonlinearity = getattr(F, nonlinearity)

    def forward(self, x):
        # x: [batch, time, n_inputs]
        return self.nonlinearity(self.linear(x))
```

**Approach B: Random Projection (Fixed)**
```python
class RandomProjection(nn.Module):
    def __init__(self, n_inputs, embed_dim):
        super().__init__()
        # Fixed random projection matrix
        W = torch.randn(n_inputs, embed_dim) / np.sqrt(n_inputs)
        self.register_buffer('W', W)

    def forward(self, x):
        return torch.relu(x @ self.W)
```

**Approach C: Time-Lagged Features**
```python
def add_time_lags(x, lags=[1, 2, 3]):
    """Add time-lagged copies of inputs."""
    # x: [batch, time, n_inputs]
    features = [x]
    for lag in lags:
        lagged = torch.zeros_like(x)
        lagged[:, lag:, :] = x[:, :-lag, :]
        features.append(lagged)
    return torch.cat(features, dim=-1)
```

**Approach D: Polynomial Features**
```python
def polynomial_features(x, degree=2):
    """Add polynomial features up to given degree."""
    features = [x]
    if degree >= 2:
        # Add pairwise products (x_i * x_j for i <= j)
        for i in range(x.shape[-1]):
            for j in range(i, x.shape[-1]):
                features.append(x[..., i:i+1] * x[..., j:j+1])
    return torch.cat(features, dim=-1)
```

### Integration with EIRNN

```python
class EIRNN(nn.Module):
    def __init__(
        self,
        ...,
        input_embed_dim: int = None,  # If set, expand inputs
        input_embed_type: str = 'learnable',  # 'learnable', 'random', 'time_lag', 'polynomial'
    ):
        if input_embed_dim is not None:
            if input_embed_type == 'learnable':
                self.input_embed = nn.Sequential(
                    nn.Linear(n_inputs, input_embed_dim),
                    nn.ReLU()
                )
                actual_n_inputs = input_embed_dim
            elif input_embed_type == 'random':
                self.input_embed = RandomProjection(n_inputs, input_embed_dim)
                actual_n_inputs = input_embed_dim
            # ... etc
        else:
            self.input_embed = None
            actual_n_inputs = n_inputs

        # Use actual_n_inputs for W_in
        self.W_in = nn.Parameter(torch.zeros(self.n_total, actual_n_inputs))
```

### Testing Configurations

```python
input_embed_experiments = [
    # Learnable expansion
    {'name': 'embed_28_learnable', 'input_embed_dim': 28, 'input_embed_type': 'learnable'},
    {'name': 'embed_56_learnable', 'input_embed_dim': 56, 'input_embed_type': 'learnable'},

    # Random projection
    {'name': 'embed_28_random', 'input_embed_dim': 28, 'input_embed_type': 'random'},
    {'name': 'embed_56_random', 'input_embed_dim': 56, 'input_embed_type': 'random'},

    # Time lags (14 * 4 = 56 dims with 3 lags)
    {'name': 'embed_time_lag', 'input_embed_type': 'time_lag', 'time_lags': [1, 2, 3]},

    # Combined with best alpha setting
    {
        'name': 'embed_28_alpha_neuron',
        'input_embed_dim': 28,
        'input_embed_type': 'learnable',
        'learnable_alpha': 'neuron',
    },
]
```

---

## Improvement 3: Fix Learnable Tau

### Problem
When `tau_i_init = 20ms` and `dt = 25ms`, the resulting `alpha_i = 25/20 = 1.25 > 1`, which breaks RNN dynamics.

### Solution
Constrain `tau_min >= dt`:

```python
# In EIRNN class
TAU_MIN = 25.0  # Changed from 10.0 - must be >= dt
TAU_MAX = 200.0

def _raw_to_tau(self, raw: torch.Tensor) -> torch.Tensor:
    """Convert raw parameter to tau via soft clamping."""
    scale = (self.TAU_MAX - self.TAU_MIN) / 2.0
    return self.TAU_MIN + torch.nn.functional.softplus(raw) * scale
```

### Testing Configuration

```python
learnable_tau_fixed = [
    {
        'name': 'learnable_tau_pop_fixed_v2',
        'learnable_tau': 'population',
        'tau_e_init': 50.0,
        'tau_i_init': 35.0,  # Changed from 20.0 to be > dt
    },
]
```

---

## Improvement 4: Hyperparameter Refinement

### Based on Learned Alpha Values

From Phase 2, the optimal alpha values were:
- E neurons: 0.54 (τ_e ≈ 46ms) - slightly slower than τ=50ms default
- I neurons: 0.72 (τ_i ≈ 35ms) - faster than τ=50ms default

### Experiments

```python
hyperparameter_experiments = [
    # Better initialization
    {
        'name': 'alpha_init_optimal',
        'learnable_alpha': 'neuron',
        'alpha_e_init': 0.54,
        'alpha_i_init': 0.72,
    },

    # Test if fixed optimal alpha matches learnable
    {
        'name': 'fixed_alpha_optimal',
        'learnable_alpha': 'none',
        'tau_e': 46.0,  # Corresponds to alpha=0.54
        'tau_i': 35.0,  # Corresponds to alpha=0.72
    },

    # Different noise scales
    {
        'name': 'noise_0.05',
        'noise_scale': 0.05,
        'learnable_alpha': 'neuron',
    },
    {
        'name': 'noise_0.15',
        'noise_scale': 0.15,
        'learnable_alpha': 'neuron',
    },
]
```

---

## Implementation Priority Order

1. **Tier 1 Combinations** (3 experiments)
   - alpha_neuron + poisson_ltrial
   - alpha_neuron + low_rank_50
   - poisson + low_rank_50

2. **Input Embedding** (4 experiments)
   - Learnable 28D and 56D
   - Time-lagged features
   - Best embedding + alpha_neuron

3. **Tier 2 Triple Combinations** (2 experiments)
   - All three improvements together
   - With activity regularization

4. **Hyperparameter Refinement** (4 experiments)
   - Optimal alpha initialization
   - Fixed optimal alpha/tau
   - Noise scale variations

5. **Fixed Learnable Tau** (1 experiment)
   - With tau_min = dt = 25ms

**Total: 14 new experiments**

---

## Experiment Script Template

Create `scripts/experiment_phase2c.py`:

```python
"""
Phase 2c Experiments - Systematic combinations and input embedding.

Usage:
    python scripts/experiment_phase2c.py --data data/rnn_export_Newton_08_15_2025_SC.mat --output results/phase2c/
"""

# Add input_embed_dim and input_embed_type to create_model_from_data
# Add alpha_e_init and alpha_i_init separate parameters
# Add support for time-lagged input features

experiments = {
    # Tier 1: Priority Combinations
    'alpha_neuron_poisson': {
        'learnable_alpha': 'neuron',
        'use_poisson_ltrial': True,
    },
    'alpha_neuron_lowrank50': {
        'learnable_alpha': 'neuron',
        'low_rank': 50,
    },
    'poisson_lowrank50': {
        'use_poisson_ltrial': True,
        'low_rank': 50,
    },

    # Tier 2: Triple Combinations
    'alpha_neuron_poisson_lowrank50': {
        'learnable_alpha': 'neuron',
        'use_poisson_ltrial': True,
        'low_rank': 50,
    },

    # Input Embedding
    'embed_28_learnable': {
        'input_embed_dim': 28,
        'input_embed_type': 'learnable',
    },
    'embed_28_alpha_neuron': {
        'input_embed_dim': 28,
        'input_embed_type': 'learnable',
        'learnable_alpha': 'neuron',
    },

    # Hyperparameter Refinement
    'alpha_init_optimal': {
        'learnable_alpha': 'neuron',
        'alpha_e_init': 0.54,
        'alpha_i_init': 0.72,
    },
}
```

---

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|--------------|
| Val PSTH Correlation | ≥ 0.3614 | 0.3449 |
| Improvement needed | +4.8% | - |

### Secondary Metrics
- Training stability (no NaN losses)
- Biological plausibility (rates 1-50 sp/s)
- Computational efficiency (< 2s/epoch on CPU)

---

## Quick Start

To continue this work:

1. **Read this document** for full context
2. **Read `specs/improvement_experiment_results_01232026.md`** for Phase 2 details
3. **Implement input embedding** in `src/model.py`
4. **Create `scripts/experiment_phase2c.py`** with priority experiments
5. **Run experiments:**
   ```bash
   python scripts/experiment_phase2c.py \
       --data data/rnn_export_Newton_08_15_2025_SC.mat \
       --output results/phase2c/ \
       --device cpu
   ```

---

## Model Architecture Reference

```
EIRNN Dynamics:
    [Optional] embedded_input = InputEmbed(u[t])
    x[t] = (1 - α) * x[t-1] + α * (W_rec @ r[t-1] + W_in @ input[t]) + noise
    r[t] = softplus(x[t]) * rate_scale + rate_baseline
    y[t] = W_out @ r_exc[t] + b_out

Where:
    - α: Per-neuron integration constant (learnable via sigmoid)
    - W_rec: Recurrent weights (optionally low-rank: U @ V^T)
    - Dale's law: E columns positive, I columns negative
    - Input embedding: Optional expansion of input dimensionality

Loss Function:
    L = L_neuron + gradient_balanced(L_trial) + L_reg [+ L_activity]

    L_neuron = (1 - correlation) + 0.1 * scale_loss + 0.05 * variance_loss
    L_trial = Sinkhorn_OT_cost(model, target, distance='euclidean' or 'poisson')
    L_reg = λ * (||W_rec||² + ||W_in||²)
    L_activity = λ_act * (mean_rate - target_mean)² [optional]
```

---

## Appendix: Data Summary

```
Dataset: data/rnn_export_Newton_08_15_2025_SC.mat
- Trials: 1043 (835 train, 208 val with 80/20 split)
- Neurons: 54 (41 excitatory, 13 inhibitory)
- Time bins: 219 at 25ms resolution (5.475s total)
- Mean firing rate: 13.57 sp/s
- Input dimensions: 14 (task variables)
```
