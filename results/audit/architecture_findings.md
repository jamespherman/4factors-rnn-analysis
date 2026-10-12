# Architecture Findings: Model Structure and Components

## Executive Summary

The best architecture uses:
- **Attention-based input embedding** (56-dim, 4 heads)
- **Learnable initial state h0** (initialized at 0.1)
- **Standard E-I RNN** with Dale's law
- **tau=50ms** time constant
- **noise_scale=0.1**

This document details all architectural experiments and their outcomes.

---

## 1. Input Embedding

### Overview
Input embedding expands the 14-dimensional input to a higher-dimensional representation before feeding to the RNN.

### Variants Tested

| Type | Dimension | Val Corr | Relative |
|------|-----------|----------|----------|
| None | 14 | 0.2901 | baseline |
| Learnable (linear+ReLU) | 28 | 0.3341 | +15.2% |
| **Learnable (linear+ReLU)** | **56** | **0.3762** | **+29.7%** |
| Learnable (linear+ReLU) | 112 | 0.3440 | +18.6% |
| Learnable (linear+ReLU) | 168 | 0.3120 | +7.5% |
| Random projection | 28 | 0.3245 | +11.8% |
| Time-lagged features | 56 | 0.3365 | +16.0% |
| Deep MLP (14→28→56) | 56 | 0.3630 | +25.1% |
| Recurrent (GRU) | 56 | 0.3733 | +28.7% |
| **Attention** | **56** | **0.3843** | **+32.5%** |
| Typed (separate per group) | 56 | 0.3305 | +13.9% |

### Key Findings

1. **4x expansion (56 dim) is optimal**
   - 2x (28 dim): Good but less effective
   - 8x (112 dim): Overfitting, performance drops
   - 12x (168 dim): Significant overfitting

2. **Attention embedding is best**
   - Self-attention over 14 input channels
   - Captures cross-input dependencies
   - +32.5% improvement over baseline

3. **Learnable beats fixed**
   - Random projection: +11.8%
   - Time-lagged (fixed): +16.0%
   - Learnable: +29.7%

### Optimal Configuration
```python
input_embed_dim = 56
input_embed_type = 'attention'
attention_heads = 4
```

---

## 2. Attention Mechanism

### Implementation
Self-attention over input features:
```python
# Each of 14 inputs treated as a token
# Multi-head attention with 4 heads
# Output: 56-dimensional embedding per timestep
```

### Parameters Tested
Only 4-head attention tested systematically. Other head counts not explored.

### Performance
| Config | Val Corr |
|--------|----------|
| attention_embedding (Phase 3) | 0.3843 |
| attention_learnable_h0 (Phase 4) | 0.4021 |
| attention + extended training | 0.4121 |

### Why It Works
- Learns which input features are relevant at each timestep
- Captures interactions between inputs (e.g., location × reward)
- More expressive than linear embedding

---

## 3. Learnable Initial State (h0)

### Motivation
Allow network to start from non-zero baseline, matching non-zero firing rates at trial onset.

### Implementation
```python
learnable_h0 = True
h0_init = 0.1  # Initial value before learning
```

### Impact
| Config | Val Corr | vs Attention Baseline |
|--------|----------|---------------------|
| Attention baseline | 0.3907 | - |
| **Attention + h0** | **0.4021** | **+2.9%** |
| Attention + h0 + AdamW | 0.3987 | +2.0% |

### Learned h0 Statistics
```
Mean: 0.291
Std: 0.204
Min: -0.422
Max: 0.603

E neurons: mean 0.26
I neurons: mean 0.39

Positive values: 92.3% of neurons
```

### Key Insight
I neurons learned higher initial states than E neurons, potentially reflecting tonic inhibition at trial onset.

---

## 4. Time Constants (tau)

### Default Configuration
```python
tau = 50.0  # ms
dt = 25.0   # ms (bin size)
alpha = dt / tau = 0.5  # Integration constant
```

### Variants Tested

| tau (ms) | alpha | Val Corr | Effect |
|----------|-------|----------|--------|
| 25 | 1.0 | 0.384 | -4.5% |
| **50** | **0.5** | **0.402** | **baseline** |

### Learnable Tau
| Config | tau_e | tau_i | Val Corr |
|--------|-------|-------|----------|
| learnable_tau_pop_v2 | 43.3ms | 35.1ms | 0.3461 |
| learnable_tau_phase5 | ~47ms | ~38ms | 0.388 |

### Findings
1. **tau=50ms is appropriate** for SC dynamics
2. Faster tau (25ms) hurt performance
3. Learnable tau converges to similar values (E: ~45ms, I: ~35ms)
4. Learnable tau doesn't improve over fixed tau with attention

### I Neurons Are Faster
Both learnable_alpha and learnable_tau show:
- E neurons: slower integration (tau ~46ms, alpha ~0.54)
- I neurons: faster integration (tau ~35ms, alpha ~0.72)

---

## 5. Learnable Alpha (Integration Constant)

### Implementation
```python
# alpha = dt/tau, but learned directly
# Using sigmoid to constrain alpha ∈ (0, 1)
```

### Variants
| Variant | Learned Values | Val Corr |
|---------|----------------|----------|
| Scalar (global) | α = 0.56 | 0.3105 |
| Population (E/I) | αE=0.58, αI=0.73 | 0.3443 |
| **Per-neuron** | **αE=0.54±0.07, αI=0.72±0.03** | **0.3449** |

### When to Use
- **Before attention:** Per-neuron alpha was best Phase 2 result
- **With attention:** Alpha didn't help (0.3789 vs 0.3907)

### Interpretation
Attention embedding implicitly captures temporal dynamics that learnable alpha was meant to address.

---

## 6. Hidden Size and E:I Ratio

### E:I Ratio Enforcement
```python
# 4:1 E:I ratio (80% E, 20% I)
# Hidden units added to maintain ratio
```

### Network Sizes
| Dataset | Recorded | Hidden | Total |
|---------|----------|--------|-------|
| 08_15 | 54 (41E/13I) | 11 (11E/0I) | 65 |
| 08_14 | 80 (25E/55I) | 100+ | ~180 |
| 08_13 | 20 (10E/10I) | varies | varies |

### Large Model Tests
| Config | Total Units | Val Corr |
|--------|-------------|----------|
| Standard | 65 | 0.407 |
| test_large_model | 200 | 0.328 |
| test_large_model_v2 | 200 | varies |

### Findings
- Larger models don't improve performance
- Standard ratio enforcement works well
- No benefit from adding more hidden units

---

## 7. Dale's Law

### Implementation
```python
# E neurons (indices 0:n_exc): positive outgoing weights
# I neurons (indices n_exc:): negative outgoing weights
W_rec = torch.abs(W_rec_raw) * sign_mask
```

### Bypass Test
| Config | Val Corr | Effect |
|--------|----------|--------|
| With Dale's law | 0.402 | baseline |
| bypass_dale | 0.383 | -4.7% |

### Conclusion
**Dale's law helps.** Removing the constraint hurts performance. The biological constraint provides useful regularization.

---

## 8. Activation Function

### Default
```python
activation = F.softplus  # log(1 + exp(x))
```

### Rationale
- Smooth (unlike ReLU)
- Always positive (firing rates can't be negative)
- Approximates ReLU for large x

### Status
No alternative activations tested.

---

## 9. Noise Scale

### Values Tested
| Noise | Val Corr | Effect |
|-------|----------|--------|
| 0.05 | 0.3377 | -2.1% |
| **0.1** | **0.402** | **baseline** |
| 0.15 | 0.350 | +1.5% (Phase 2c) |
| 0.3 | 0.385 | -4.2% (Phase 5) |
| 0.5 | 0.384 | -4.5% (Phase 5) |

### Findings
- **noise_scale=0.1 is optimal**
- Slightly higher (0.15) showed marginal benefit in Phase 2c
- Much higher (0.3, 0.5) hurt significantly
- Noise doesn't help match trial-to-trial variability (Fano factor still 17x lower than data)

---

## 10. Spectral Radius

### Configuration
```python
spectral_radius = 0.9  # Target for W_rec initialization
```

### Status
- Fixed at 0.9 in all experiments
- No systematic exploration of other values
- 0.9 provides stable dynamics without chaotic behavior

---

## 11. Low-Rank Connectivity

### Motivation
Constrain W_rec = U @ V^T for regularization and reduced parameters.

### Results
| Rank | Val Corr | Effect |
|------|----------|--------|
| 10 | 0.3167 | +9.2% vs baseline |
| 20 | 0.3227 | +11.2% |
| 30 | 0.3120 | +7.6% |
| **50** | **0.3334** | **+14.9%** |
| Full (65) | 0.2901 | baseline |

### With Best Config (Phase 5)
```
low_rank_50_phase5: 0.399 (vs 0.402 baseline)
```

### Findings
- Low-rank provides regularization benefit
- Rank=50 best in Phase 2
- Near-baseline with best config (viable alternative)
- Non-monotonic relationship with rank

---

## Summary: Optimal Architecture

```python
# Input processing
input_embed_dim = 56
input_embed_type = 'attention'
attention_heads = 4

# Initial state
learnable_h0 = True
h0_init = 0.1

# Dynamics
tau = 50.0
noise_scale = 0.1
spectral_radius = 0.9

# Structure
enforce_dale = True  # Dale's law
enforce_ratio = True  # 4:1 E:I

# Activation
activation = 'softplus'
```

---

## Untested Architectural Variants

1. **Multi-layer RNNs** - Only single layer tested
2. **Bidirectional processing** - Not applicable to online prediction
3. **Different attention configurations** - Only 4 heads tested
4. **Skip connections** - Not explored
5. **Layer normalization** - Not tested
6. **Multi-timescale RNNs** - Suggested in Phase 5 for high-frequency content
7. **Separate E/I dynamics** - Different equations for E vs I neurons
8. **Alternative activations** - Only softplus tested
