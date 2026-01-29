# E-I RNN Improvement Plan - January 23, 2026

## Current Baseline

**Best validation PSTH correlation: 0.3614** (9.5% improvement over no-L_trial baseline of 0.3299)

### Current Configuration
- Gradient balancing with ltrial_scale=0.5
- Sinkhorn optimal transport (epsilon=0.1, iterations=20)
- MSE-based L_neuron loss (correlation + scale + variance)
- Single time constant τ=50ms for all neurons
- Full-rank recurrent connectivity with Dale's law
- Adam optimizer, lr=1e-3, ReduceLROnPlateau scheduler
- No learning rate warmup

---

## Proposed Improvements

Based on literature review and the suggestions provided, I propose testing the following improvements in order of expected impact and implementation complexity:

### Phase 1: Loss Function Improvements (High Priority)

#### 1.1 Poisson Log-Likelihood Loss
**Rationale**: Spike counts follow Poisson statistics where variance equals mean. MSE assumes homoscedastic errors, which is incorrect for neural data. Research shows Poisson-based likelihoods better capture spike statistics ([biorxiv](https://www.biorxiv.org/content/10.1101/2020.07.13.201673v1.full)).

**Implementation**:
```python
def poisson_loss(model_rates, target_rates, eps=1e-8):
    """Poisson negative log-likelihood loss."""
    # L = model_rates - target_rates * log(model_rates + eps)
    return (model_rates - target_rates * torch.log(model_rates + eps)).mean()
```

**Key considerations**:
- Target rates must be non-negative (already true for spike data)
- Model rates must be positive (softplus ensures this)
- More stable gradient flow for low firing rate neurons
- May need to combine with correlation loss for shape matching

**Testing**: Compare Poisson vs MSE vs hybrid (Poisson + correlation) losses.

#### 1.2 Activity Regularization
**Rationale**: Penalize unrealistic firing rates to keep network in biologically plausible regime. Very high or very low rates indicate the model is overfitting to noise.

**Implementation**:
```python
def activity_regularization(rates, target_mean=10.0, lambda_act=0.01):
    """Penalize deviation from target mean firing rate."""
    mean_rate = rates.mean()
    return lambda_act * (mean_rate - target_mean) ** 2
```

**Testing**: Sweep lambda_act from 0.001 to 0.1.

---

### Phase 2: Architecture Improvements (Medium Priority)

#### 2.1 Learnable Time Constants (Per-Population)
**Rationale**: Real neurons have diverse time constants. Interneurons typically have faster dynamics (τ~10-20ms) than excitatory neurons (τ~40-80ms). Research from [Nature Scientific Reports](https://www.nature.com/articles/s41598-020-68169-x) shows learnable time constants improve RNN performance and memory capacity.

**Implementation** (conservative - per-population):
```python
# In EIRNN.__init__:
self.tau_e = nn.Parameter(torch.tensor(50.0))  # E time constant
self.tau_i = nn.Parameter(torch.tensor(20.0))  # I time constant (faster)

# In forward pass:
alpha_e = self.dt / self.tau_e
alpha_i = self.dt / self.tau_i
alpha = torch.cat([alpha_e.expand(self.n_exc), alpha_i.expand(self.n_inh)])
```

**Constraints**: Clamp τ to [10, 200]ms to prevent instability.

**Testing**: Compare fixed τ=50ms vs learnable per-population vs learnable per-neuron.

#### 2.2 Low-Rank Recurrent Connectivity
**Rationale**: Constrain W_rec = UV^T where U,V are [n_neurons × rank]. Acts as strong regularizer, improves interpretability, and can improve generalization ([PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010426)).

**Implementation**:
```python
# Replace W_rec_raw with low-rank factors
self.U = nn.Parameter(torch.randn(n_total, rank) / np.sqrt(rank))
self.V = nn.Parameter(torch.randn(n_total, rank) / np.sqrt(rank))

@property
def W_rec(self):
    W_low_rank = self.U @ self.V.T
    # Apply Dale's law and diagonal mask
    return torch.abs(W_low_rank) * self.sign_mask * self.diag_mask
```

**Testing**: Sweep rank from 5 to 50 (network has ~200 neurons).

---

### Phase 3: Training Dynamics (Medium Priority)

#### 3.1 Learning Rate Warmup + Cosine Annealing
**Rationale**: Warmup stabilizes early training by allowing optimizer statistics (Adam's momentum) to adapt before using aggressive learning rates. Cosine annealing provides smoother decay than plateau-based scheduling ([Overview](https://www.gaohongnan.com/playbook/training/why_cosine_annealing_warmup_stabilize_training.html)).

**Implementation**:
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Testing**:
- Warmup: 50-100 epochs (10-20% of training)
- Compare cosine vs ReduceLROnPlateau

**Note**: Previous experiments showed curriculum warmup (delaying L_trial) hurt performance. LR warmup is different - it affects optimizer dynamics, not loss weighting.

---

### Phase 4: Data Augmentation (Lower Priority)

#### 4.1 Input Noise Injection
**Rationale**: Add small Gaussian noise to continuous inputs (eye position) during training to improve robustness and prevent overfitting to exact input trajectories.

**Implementation**:
```python
# During training only
if training:
    # Add noise to continuous inputs (indices 7-8 are eye position)
    noise = torch.randn_like(inputs[:, :, 7:9]) * noise_std
    inputs[:, :, 7:9] = inputs[:, :, 7:9] + noise
```

**Testing**: Sweep noise_std from 0.05 to 0.2 (inputs are z-scored).

---

## Experiment Plan

### Experiment 1: Poisson Loss Variants
Test 3 configurations against baseline:
1. **Pure Poisson**: Replace L_neuron with Poisson NLL
2. **Poisson + Correlation**: Poisson for magnitude, correlation for shape
3. **Hybrid**: 0.5 * Poisson + 0.5 * current L_neuron

Expected runtime: ~3 runs × 500 epochs = moderate

### Experiment 2: Learnable Time Constants
Test 2 configurations:
1. **Per-population τ**: τ_e, τ_i learnable
2. **Per-neuron τ**: Full flexibility (may overfit)

Expected runtime: ~2 runs × 500 epochs = moderate

### Experiment 3: Training Schedule
Test 3 configurations:
1. **LR Warmup**: 50 epoch warmup + ReduceLROnPlateau
2. **Cosine Annealing**: No warmup, cosine schedule
3. **Warmup + Cosine**: 50 epoch warmup + cosine decay

Expected runtime: ~3 runs × 500 epochs = moderate

### Experiment 4: Combination of Best Approaches
Combine winners from Experiments 1-3 into final configuration.

---

## Implementation Order

1. **First**: Implement Poisson loss option (lowest code change, potentially high impact)
2. **Second**: Implement learnable time constants (medium code change, good theoretical basis)
3. **Third**: Implement cosine annealing scheduler (low code change, easy to test)
4. **Fourth**: Implement low-rank connectivity option (higher code change, strong regularizer)
5. **Fifth**: Implement activity regularization and input noise (simple additions)

---

## Files to Modify

1. `src/losses.py` - Add Poisson loss functions, activity regularization
2. `src/model.py` - Add learnable time constants, low-rank option
3. `scripts/train_model.py` - Add cosine scheduler, noise injection, new CLI args
4. `scripts/experiment_improvements.py` - New experiment script (to be created)

---

## Success Metrics

- **Primary**: Validation PSTH correlation (target: >0.38, ~5% improvement)
- **Secondary**: Training stability, convergence speed
- **Constraints**: Biologically plausible firing rates (1-50 sp/s mean)

---

## Risk Assessment

| Improvement | Expected Impact | Implementation Risk | Notes |
|------------|----------------|---------------------|-------|
| Poisson Loss | Medium-High | Low | Standard for neural data |
| Learnable τ | Medium | Low | Well-established approach |
| Cosine Annealing | Low-Medium | Low | Easy to implement |
| Low-Rank W_rec | Medium | Medium | May interact with Dale's law |
| Activity Reg | Low | Low | Simple addition |
| Input Noise | Low | Low | Standard augmentation |

---

## References

1. [Neural Network Poisson Models for Behavioural and Neural Spike Train Data](https://www.biorxiv.org/content/10.1101/2020.07.13.201673v1.full) - bioRxiv
2. [Adaptive time scales in recurrent neural networks](https://www.nature.com/articles/s41598-020-68169-x) - Scientific Reports
3. [The impact of sparsity in low-rank recurrent neural networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010426) - PLOS Comp Bio
4. [Why Does Cosine Annealing With Warmup Stabilize Training?](https://www.gaohongnan.com/playbook/training/why_cosine_annealing_warmup_stabilize_training.html) - Gaohongnan
5. [Extracting computational mechanisms from neural data using low-rank RNNs](https://pillowlab.princeton.edu/pubs/Valente22neurips_lowrankRNNs.pdf) - NeurIPS

---

## Experimental Results (January 23, 2026)

### Experiments Conducted

The following experiments were run on the Newton_08_15_2025_SC dataset:

| Configuration | Best Val Corr | Epochs | Notes |
|--------------|---------------|--------|-------|
| **baseline** | **0.2901** | 500 | Reference implementation with gradient balancing |
| poisson_loss | 0.1730 | ~500 | Pure Poisson NLL loss |
| hybrid_loss_0.3 | 0.1499 | 124 | 30% Poisson + 70% correlation |
| hybrid_loss_0.5 | 0.1509 | 121 | 50% Poisson + 50% correlation |
| learnable_tau_population | 0.0255 | 287 | Per-population τ (E/I) |
| learnable_tau_neuron | 0.0082 | 100 | Per-neuron τ |
| cosine_no_warmup | In progress | - | Cosine annealing, no warmup |
| cosine_warmup_50 | Not run | - | Cosine annealing with 50 epoch warmup |
| activity_reg_0.01 | Not run | - | Activity regularization |

### Key Findings

#### 1. Poisson Loss Performs Worse Than Correlation Loss

**Result**: Poisson-based losses significantly underperformed the correlation-based baseline.
- Pure Poisson: 0.1730 vs baseline 0.2901 (-40%)
- Hybrid 50%: 0.1509 vs baseline (-48%)

**Analysis**: While Poisson loss is theoretically better suited for spike count data, it appears that:
- The correlation loss captures temporal dynamics better for PSTH matching
- Poisson loss may over-emphasize magnitude matching at the expense of shape
- The current data preprocessing (smoothing, z-scoring) may already normalize variance

**Recommendation**: Keep correlation-based loss as primary. Consider Poisson as a secondary term with very low weight (λ < 0.1) if revisiting.

#### 2. Learnable Time Constants Cause Catastrophic Failure

**Result**: Making time constants learnable destroyed model performance.
- Population τ: 0.0255 (91% worse than baseline)
- Per-neuron τ: 0.0082 (97% worse than baseline)

**Analysis**: Examination of learned τ values shows they barely changed from initialization (τ_e ≈ 50ms, τ_i ≈ 20ms), yet the model failed. This suggests:
- The issue is in the implementation's interaction with the RNN dynamics
- Per-neuron alpha values may break the numerical stability of Euler integration
- The noise scaling or state update equations need adjustment for non-scalar alpha

**Recommendation**: The learnable tau feature requires debugging before it can be useful. The issue is likely in how per-neuron alpha is applied in the forward pass.

#### 3. Baseline Discrepancy

**Observation**: Our baseline achieved 0.2901 vs the reported 0.3614 for the best configuration.

**Possible causes**:
- Different random seed (42 vs original experiments)
- Missing gradient balancing implementation details from original
- Different data splits

**Recommendation**: Verify the gradient balancing implementation matches the original exactly.

### Files Modified

1. `src/losses.py`:
   - Added `compute_L_poisson()` - Poisson negative log-likelihood
   - Added `compute_L_neuron_hybrid()` - Combined Poisson + correlation
   - Added `compute_activity_regularization()` - Rate regularization
   - Added `get_cosine_schedule_with_warmup()` - LR scheduler

2. `src/model.py`:
   - Added `learnable_tau` parameter ('none', 'population', 'neuron')
   - Added `tau_e_init`, `tau_i_init` parameters
   - Added `get_alpha()` method for per-neuron integration constants
   - Added `get_tau_values()` method for logging
   - Updated `forward()` to use per-neuron alpha when applicable

3. `scripts/experiment_improvements.py`:
   - New experiment script testing all improvements

### Next Steps

1. **Debug learnable tau**: Investigate why per-neuron alpha breaks training
2. **Test cosine annealing**: Complete remaining experiments
3. **Consider low-rank connectivity**: This approach wasn't tested but has strong theoretical basis
4. **Investigate baseline gap**: Understand why our baseline underperforms reported results
5. **Try smaller tau learning rate**: Use separate optimizer with lower LR for tau parameters

### Conclusion

The original correlation-based loss with gradient balancing remains the best approach for this task. Alternative loss functions (Poisson, hybrid) did not improve and may hurt performance. Learnable time constants require implementation fixes before they can be properly evaluated.

**Current Best**: Correlation loss + gradient balancing (ltrial_scale=0.5, sinkhorn_epsilon=0.1)
