# E-I RNN Improvement Plan - Phase 2
## January 23, 2026

This document provides a complete context for continuing optimization work on the E-I RNN model for SC neural data fitting.

---

## Project Overview

### Goal
Train an Excitatory-Inhibitory RNN to fit neural recordings from Superior Colliculus (SC), matching both:
1. **Trial-averaged activity (PSTH)** - Mean firing rates across trials
2. **Single-trial variability** - Trial-to-trial differences in population trajectories

### Current Best Result
- **Validation PSTH correlation: 0.3614** (from prior experiments documented in `specs/current_best_config.md`)
- Configuration: Gradient balancing, ltrial_scale=0.5, Sinkhorn epsilon=0.1

### Key Files
- `src/model.py` - EIRNN class with Dale's law constraints
- `src/losses.py` - Loss functions (L_neuron, L_trial with Sinkhorn, L_reg)
- `src/data_loader.py` - Data loading and preprocessing
- `scripts/train_model.py` - Main training script
- `scripts/experiment_improvements.py` - Experiment script (created in Phase 1)

### Data
- Location: `data/rnn_export_Newton_08_15_2025_SC.mat`
- 1043 trials, 54 neurons (41 E, 13 I), 219 time bins at 25ms

---

## Phase 1 Results Summary

The following improvements were tested on January 23, 2026:

| Configuration | Val Correlation | Result |
|--------------|-----------------|--------|
| baseline (correlation loss) | 0.2901 | Reference |
| poisson_loss (for L_neuron) | 0.1730 | -40% worse |
| hybrid_loss_0.3 | 0.1499 | -48% worse |
| hybrid_loss_0.5 | 0.1509 | -48% worse |
| learnable_tau_population | 0.0255 | -91% worse (BROKEN) |
| learnable_tau_neuron | 0.0082 | -97% worse (BROKEN) |

### Key Learnings
1. **Poisson loss for L_neuron hurts performance** - Correlation-based loss is better for PSTH shape matching
2. **Learnable time constants implementation is broken** - Needs debugging
3. **Baseline gap exists** - Our 0.2901 vs reported 0.3614 needs investigation

### Code Added in Phase 1
In `src/losses.py`:
- `compute_L_poisson()` - Poisson NLL loss
- `compute_L_neuron_hybrid()` - Combined Poisson + correlation
- `compute_activity_regularization()` - Rate regularization
- `get_cosine_schedule_with_warmup()` - Cosine LR scheduler

In `src/model.py`:
- `learnable_tau` parameter ('none', 'population', 'neuron')
- `get_alpha()` method for per-neuron integration constants
- `get_tau_values()` method for logging

---

## Phase 2 Plan: Five Improvements to Address

### Improvement 1: Poisson Loss for L_trial (Not L_neuron)

**Rationale**: While PSTH (trial-averaged) rates benefit from correlation-based loss for shape matching, the **trial-to-trial variability** in spike counts follows Poisson statistics. Using Poisson loss for L_trial may better capture this natural variance structure.

**Current L_trial Implementation** (`src/losses.py:194-268`):
```python
def compute_L_trial(...):
    # Population-average activity per trial
    model_pop = model_rates.mean(dim=2)   # [batch, time]
    target_pop = target_rates.mean(dim=2)

    # Z-score normalize across trials
    # Compute pairwise Euclidean distances
    # Apply Sinkhorn optimal transport
    # Return weighted distance
```

**Proposed Change**: Replace Euclidean distance with Poisson-based distance in the trial matching:

```python
def compute_L_trial_poisson(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    ...
    use_poisson_distance: bool = True
) -> torch.Tensor:
    """
    Trial-matching loss with Poisson-based distance metric.

    Instead of Euclidean distance between z-scored trajectories,
    compute Poisson divergence: sum(model - target * log(model + eps))

    This respects the natural variance structure of spike counts.
    """
    # Population-average activity per trial
    model_pop = model_rates.mean(dim=2)   # [batch, time]
    target_pop = target_rates.mean(dim=2)

    # Temporal smoothing (same as before)
    kernel_size = max(1, int(smooth_ms / bin_size_ms))
    model_pop = smooth_temporal(model_pop, kernel_size, dim=1)
    target_pop = smooth_temporal(target_pop, kernel_size, dim=1)

    if use_poisson_distance:
        # Poisson divergence distance matrix
        # For each pair (i, j): sum_t [model_i(t) - target_j(t) * log(model_i(t) + eps)]
        eps = 1e-8
        model_expanded = model_pop.unsqueeze(1)  # [batch, 1, time]
        target_expanded = target_pop.unsqueeze(0)  # [1, batch, time]

        # Poisson NLL per time point, summed over time
        poisson_div = model_expanded - target_expanded * torch.log(model_expanded + eps)
        distances = poisson_div.sum(dim=2)  # [batch, batch]
    else:
        # Original Euclidean distance on z-scored trajectories
        # ... existing code ...

    # Sinkhorn optimal transport (same as before)
    P = sinkhorn_assignment(distances, n_iters=sinkhorn_iters, epsilon=sinkhorn_epsilon)

    n_trials = distances.shape[0]
    L_trial = (P * distances).sum() / n_trials

    return L_trial
```

**Testing Plan**:
1. Add `use_poisson_distance` flag to `compute_L_trial()`
2. Test with baseline L_neuron (correlation) + Poisson L_trial
3. Compare against baseline L_neuron + Euclidean L_trial

---

### Improvement 2: Learnable Mixing Parameter (Alpha)

**Rationale**: Currently, the integration constant α = dt/τ is derived from the time constant. Making α directly learnable (per-neuron or per-population) may be more stable than learning τ, and gives the optimizer direct control over the mixing between current state and new input.

**Current Dynamics**:
```
x[t] = (1 - α) * x[t-1] + α * (W_rec @ r[t-1] + W_in @ u[t]) + noise
```
where α = dt/τ ≈ 0.5 for dt=25ms, τ=50ms.

**Proposed Implementation**:
```python
class EIRNN(nn.Module):
    def __init__(self, ..., learnable_alpha: str = 'none', alpha_init: float = 0.5):
        """
        learnable_alpha: 'none', 'scalar', 'population', or 'neuron'
        alpha_init: Initial value (0.5 corresponds to τ=50ms with dt=25ms)
        """
        if learnable_alpha == 'none':
            self.alpha = dt / tau  # Fixed
        elif learnable_alpha == 'scalar':
            # Single learnable alpha for entire network
            self._alpha_logit = nn.Parameter(torch.tensor(self._alpha_to_logit(alpha_init)))
        elif learnable_alpha == 'population':
            # Separate alpha for E and I populations
            self._alpha_e_logit = nn.Parameter(torch.tensor(self._alpha_to_logit(alpha_init)))
            self._alpha_i_logit = nn.Parameter(torch.tensor(self._alpha_to_logit(alpha_init * 1.5)))  # I faster
        elif learnable_alpha == 'neuron':
            # Per-neuron alpha
            alpha_init_tensor = torch.full((n_total,), self._alpha_to_logit(alpha_init))
            self._alpha_logit = nn.Parameter(alpha_init_tensor)

    @staticmethod
    def _alpha_to_logit(alpha):
        """Convert alpha in (0, 1) to unconstrained logit space."""
        alpha = np.clip(alpha, 0.01, 0.99)
        return np.log(alpha / (1 - alpha))

    @staticmethod
    def _logit_to_alpha(logit):
        """Convert logit to alpha in (0, 1) via sigmoid."""
        return torch.sigmoid(logit)

    def get_alpha(self, device):
        """Get constrained alpha values."""
        if self.learnable_alpha == 'none':
            return torch.tensor(self.alpha, device=device)
        elif self.learnable_alpha == 'scalar':
            return self._logit_to_alpha(self._alpha_logit)
        elif self.learnable_alpha == 'population':
            alpha_e = self._logit_to_alpha(self._alpha_e_logit)
            alpha_i = self._logit_to_alpha(self._alpha_i_logit)
            return torch.cat([alpha_e.expand(self.n_exc), alpha_i.expand(self.n_inh)])
        elif self.learnable_alpha == 'neuron':
            return self._logit_to_alpha(self._alpha_logit)
```

**Key Design Decisions**:
- Use **logit parameterization** to keep alpha in (0, 1) without clipping
- This is more stable than learning τ directly
- Sigmoid ensures smooth, bounded values

**Testing Plan**:
1. Test `learnable_alpha='scalar'` first (simplest)
2. Then test `learnable_alpha='population'`
3. Compare learned alpha values to baseline (α=0.5)

---

### Improvement 3: Debug Learnable Time Constants

**Problem**: The `learnable_tau` feature (added in Phase 1) causes training to fail catastrophically, even though the τ values barely change from initialization.

**Symptoms**:
- `learnable_tau_population`: 0.0255 correlation (vs 0.2901 baseline)
- `learnable_tau_neuron`: 0.0082 correlation
- Learned τ values: τ_e ≈ 50ms, τ_i ≈ 20ms (essentially unchanged)

**Hypothesis**: The issue is NOT the τ values themselves, but how per-neuron alpha is computed and applied in the forward pass.

**Current Implementation** (`src/model.py`):
```python
def get_alpha(self, device):
    if self.learnable_tau == 'population':
        tau_e = torch.clamp(self._tau_e, min=10.0, max=200.0)
        tau_i = torch.clamp(self._tau_i, min=10.0, max=200.0)
        alpha_e = self.dt / tau_e
        alpha_i = self.dt / tau_i
        alpha = torch.cat([alpha_e.expand(self.n_exc), alpha_i.expand(self.n_inh)])
        return alpha
```

**Potential Issues to Debug**:

1. **Gradient flow through clamp**: `torch.clamp` may have zero gradients at boundaries
   - Fix: Use soft clamping via sigmoid or softplus

2. **Alpha tensor shape in forward pass**:
   ```python
   x = (1 - alpha) * x + alpha * (rec_input + ext_input) + noise
   ```
   When alpha is [n_total], this should broadcast correctly, but verify shapes.

3. **Noise scaling inconsistency**:
   ```python
   alpha_for_noise = alpha.mean()  # Using mean alpha for noise
   noise = self.noise_scale * torch.randn_like(x) * (alpha_for_noise ** 0.5)
   ```
   This may be fine, but verify it's not causing issues.

4. **Device mismatch**: Ensure all tensors are on the same device.

**Debugging Plan**:
1. Add print statements to verify tensor shapes in forward pass
2. Test with `learnable_tau='population'` but **freeze tau parameters** (set requires_grad=False)
   - If this still fails, the issue is in how alpha is applied
   - If this works, the issue is in gradient flow through tau
3. Replace `torch.clamp` with softplus-based soft clamping
4. Test Improvement 2 (learnable alpha) as an alternative approach

---

### Improvement 4: Low-Rank Recurrent Connectivity

**Rationale**: Constrain W_rec = U @ V^T where U, V are [n_neurons × rank]. This:
- Acts as strong regularizer (fewer parameters)
- Improves interpretability (low-dimensional dynamics)
- May improve generalization

**References**:
- [The impact of sparsity in low-rank RNNs](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010426)
- [Extracting computational mechanisms from neural data using low-rank RNNs](https://pillowlab.princeton.edu/pubs/Valente22neurips_lowrankRNNs.pdf)

**Implementation**:
```python
class EIRNN(nn.Module):
    def __init__(self, ..., low_rank: int = None):
        """
        low_rank: If specified, constrain W_rec to this rank.
                  Typical values: 5-50 for ~200 neuron networks.
        """
        self.low_rank = low_rank

        if low_rank is not None:
            # Low-rank factorization: W_rec = U @ V^T
            self.U = nn.Parameter(torch.randn(self.n_total, low_rank) / np.sqrt(low_rank))
            self.V = nn.Parameter(torch.randn(self.n_total, low_rank) / np.sqrt(low_rank))
            self.W_rec_raw = None  # Not used
        else:
            # Full-rank (existing implementation)
            self.W_rec_raw = nn.Parameter(torch.zeros(self.n_total, self.n_total))
            self.U = None
            self.V = None

    @property
    def W_rec(self):
        if self.low_rank is not None:
            # Compute low-rank W_rec
            W_low_rank = self.U @ self.V.T

            if self.bypass_dale:
                W = W_low_rank * self.diag_mask
            else:
                # Apply Dale's law: take absolute value, then apply sign mask
                W = torch.abs(W_low_rank) * self.sign_mask * self.diag_mask
            return W
        else:
            # Existing full-rank implementation
            if self.bypass_dale:
                W = self.W_rec_raw * self.diag_mask
            else:
                W = torch.abs(self.W_rec_raw) * self.sign_mask * self.diag_mask
            return W
```

**Considerations**:
- Low-rank + Dale's law: Taking absolute value of low-rank matrix may reduce effective rank
- Alternative: Apply sign mask to U or V, not the product
- Start with rank ~20-30 (roughly sqrt of n_neurons)

**Testing Plan**:
1. Test rank=10, 20, 30, 50
2. Compare against full-rank baseline
3. Examine learned U, V for interpretability

---

### Improvement 5: Complete Remaining Experiments

The following experiments from Phase 1 were not completed:

#### 5a. Cosine Annealing with Warmup

**Implementation** (already in `src/losses.py`):
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,   # 50 epoch warmup
    num_training_steps=500  # Total epochs
)
```

**Testing Configurations**:
- `cosine_no_warmup`: Cosine annealing, warmup=0
- `cosine_warmup_50`: Cosine annealing, warmup=50 epochs

#### 5b. Activity Regularization

**Implementation** (already in `src/losses.py`):
```python
L_activity = compute_activity_regularization(
    model_rates,
    target_mean=10.0,   # Target mean firing rate (sp/s)
    target_max=100.0,   # Max acceptable rate
    lambda_mean=0.01,   # Weight for mean penalty
    lambda_max=0.001    # Weight for max penalty
)
```

**Testing Configurations**:
- `activity_reg_0.01`: lambda_mean=0.01
- `activity_reg_0.001`: lambda_mean=0.001

---

## Implementation Priority Order

1. **Improvement 3 (Debug learnable tau)** - Understand what's broken before adding more complexity
2. **Improvement 2 (Learnable alpha)** - Alternative to tau that may be more stable
3. **Improvement 1 (Poisson L_trial)** - Quick to implement, good theoretical basis
4. **Improvement 5 (Remaining experiments)** - Complete cosine annealing and activity reg tests
5. **Improvement 4 (Low-rank)** - Larger change, save for last

---

## Experiment Script Template

Update `scripts/experiment_improvements.py` to include new configurations:

```python
experiments = [
    # Baseline
    {'name': 'baseline', 'loss_type': 'correlation', ...},

    # Improvement 1: Poisson L_trial
    {'name': 'poisson_ltrial', 'use_poisson_ltrial': True, ...},

    # Improvement 2: Learnable alpha
    {'name': 'learnable_alpha_scalar', 'learnable_alpha': 'scalar', ...},
    {'name': 'learnable_alpha_population', 'learnable_alpha': 'population', ...},

    # Improvement 3: Debug tau (with frozen parameters)
    {'name': 'learnable_tau_frozen', 'learnable_tau': 'population', 'freeze_tau': True, ...},

    # Improvement 4: Low-rank
    {'name': 'low_rank_20', 'low_rank': 20, ...},
    {'name': 'low_rank_30', 'low_rank': 30, ...},

    # Improvement 5: Remaining
    {'name': 'cosine_warmup_50', 'scheduler': 'cosine', 'warmup_epochs': 50, ...},
    {'name': 'activity_reg', 'use_activity_reg': True, 'lambda_activity': 0.01, ...},
]
```

---

## Success Criteria

- **Primary**: Beat baseline validation correlation (0.2901 from our experiments, 0.3614 reported)
- **Secondary**: Achieve stable training with new features
- **Tertiary**: Maintain biologically plausible dynamics (rates 1-50 sp/s)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/losses.py` | Add `use_poisson_distance` to `compute_L_trial()` |
| `src/model.py` | Add `learnable_alpha` option; fix `learnable_tau`; add `low_rank` option |
| `scripts/experiment_improvements.py` | Add new experiment configurations |

---

## Quick Start for Fresh Context

To continue this work:

1. **Read this document** for full context
2. **Read `specs/current_best_config.md`** for original best configuration details
3. **Check `results/improvements_test/`** for Phase 1 experiment results
4. **Start with Improvement 3** (debugging learnable tau) to understand the codebase
5. **Run experiments** via: `python scripts/experiment_improvements.py --data data/rnn_export_Newton_08_15_2025_SC.mat --output results/improvements_phase2/`

---

## Appendix: Model Architecture Summary

```
EIRNN Dynamics:
    x[t] = (1 - α) * x[t-1] + α * (W_rec @ r[t-1] + W_in @ u[t]) + noise
    r[t] = softplus(x[t]) * rate_scale + rate_baseline
    y[t] = W_out @ r_exc[t] + b_out

Where:
    - α = dt/τ (integration constant, ~0.5 for dt=25ms, τ=50ms)
    - W_rec: Recurrent weights with Dale's law (E columns +, I columns -)
    - softplus: Ensures non-negative firing rates
    - rate_scale, rate_baseline: Learnable output scaling

Loss Function:
    L = L_neuron + gradient_balanced(L_trial) + L_reg

    L_neuron = (1 - correlation) + 0.1 * scale_loss + 0.05 * variance_loss
    L_trial = Sinkhorn_OT_cost(model_trajectories, target_trajectories)
    L_reg = λ * (||W_rec||² + ||W_in||²)
```
