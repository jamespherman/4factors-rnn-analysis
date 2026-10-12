# What Worked: Techniques That Improved Fit Quality

## Executive Summary

The following techniques demonstrated clear improvements to validation PSTH correlation. The best configuration combines **attention-based input embedding**, **learnable initial state (h0)**, **gradient balancing**, and **extended training**, achieving a peak validation correlation of **0.4121**.

---

## 1. Attention-Based Input Embedding (+32.5% from baseline)

**Evidence:**
- Phase 3 `attention_embedding`: 0.3843 vs Phase 2 baseline 0.2901
- Best of all embedding variants tested

**Why It Works:**
Self-attention over the 14 input channels allows the model to learn which input features are most relevant at each timestep. Treating each input as a token and applying multi-head attention captures cross-input dependencies that simple linear embeddings miss.

**Configuration:**
```python
input_embed_dim: 56
input_embed_type: 'attention'
attention_heads: 4
```

**Conditions:**
- Works best with 4x expansion (56 dim for 14 inputs)
- 8x and 12x expansion hurt performance (overfitting)

---

## 2. Learnable Initial State h0 (+4.6% from attention baseline)

**Evidence:**
- Phase 4 `attention_learnable_h0`: 0.4021 vs `attention_baseline`: 0.3907
- This was the modification that achieved the 0.40 target

**Why It Works:**
A learned initial hidden state allows the network to start from a non-zero baseline, better matching the non-zero firing rates observed at trial onset. The learned h0 values show:
- Mean: 0.291
- I neurons learned higher h0 (0.39) than E neurons (0.26)
- Mostly positive values (92.3%)

**Configuration:**
```python
learnable_h0: True
h0_init: 0.1  # Initial value before learning
```

**Conditions:**
- Combining h0 with AdamW slightly hurt (0.3987 vs 0.4021)
- h0 alone is sufficient; no need for additional regularization

---

## 3. Gradient Balancing (+9.5% from L_trial baseline)

**Evidence:**
- L_trial v2 `gradbal_scale_0.5`: 0.3614 vs `baseline_no_ltrial`: 0.3299
- All top-7 L_trial configurations used gradient balancing
- Fixed lambda weighting consistently hurt performance

**Why It Works:**
Gradient balancing normalizes the gradient contributions from L_neuron and L_trial to have comparable magnitudes, preventing one loss from dominating. Without it, the trial-matching loss overwhelms PSTH matching.

**Configuration:**
```python
use_grad_balancing: True
ltrial_scale: 0.5  # Scale L_trial contribution to 0.5x
```

**Conditions:**
- Scale of 0.5 works best (0.25 and 2.0 were worse)
- Must use tight Sinkhorn epsilon (0.05-0.1)

---

## 4. Input Embedding (Linear) (+29.7% from baseline)

**Evidence:**
- Phase 2c `embed_56_learnable`: 0.3762 vs baseline 0.2901
- First configuration to exceed the 0.3614 target

**Why It Works:**
Cover's theorem: expanding inputs to a higher-dimensional space via nonlinear transformation makes complex patterns more linearly separable. The 4x expansion (14 → 56) with ReLU provides optimal trade-off.

**Configuration:**
```python
input_embed_dim: 56
input_embed_type: 'learnable'  # Linear + ReLU
```

**Conditions:**
- 4x expansion optimal (2x was +15%, 8x and 12x hurt)
- Attention embedding later superseded this

---

## 5. Extended Training (+2.5% from Phase 4 best)

**Evidence:**
- Phase 5 `extended_training` (1000 epochs): 0.4121 vs Phase 4 best: 0.4021
- Final model (2000 epochs): 0.4070 (slightly lower, but more stable)

**Why It Works:**
The model was still learning at 500 epochs. Extended training allows continued refinement, especially of the learned h0 values which evolved significantly (mean increased from 0.29 to 0.51).

**Configuration:**
```python
max_epochs: 1000-2000
patience: 200-300  # Longer patience for extended training
```

**Conditions:**
- Diminishing returns after ~1000 epochs
- Best epoch for final model was 959, suggesting convergence

---

## 6. Tight Sinkhorn Epsilon (+5% relative)

**Evidence:**
- `gradbal_eps_0.1`: 0.3434 vs `gradbal_eps_0.5`: 0.3040
- `gradbal_eps_0.05`: 0.3559 (near optimal)

**Why It Works:**
Lower epsilon produces harder (more deterministic) trial-to-trial assignments in optimal transport, better capturing the bijective matching needed for trial alignment.

**Configuration:**
```python
sinkhorn_epsilon: 0.1  # or 0.05
```

**Conditions:**
- Epsilon >= 0.5 consistently hurt
- Combined with gradient balancing for best effect

---

## 7. Learnable Alpha (E/I-specific) (+18.9% from baseline)

**Evidence:**
- Phase 2 `learnable_alpha_neuron`: 0.3449 vs baseline 0.2901
- Best Phase 2 result before input embedding

**Why It Works:**
Optimal dynamics require E neurons to integrate more slowly (α ≈ 0.54) and I neurons to integrate faster (α ≈ 0.72) than the default assumption. Per-neuron learning captures this heterogeneity.

**Learned Values:**
- Excitatory: α = 0.54 ± 0.07 (τ ≈ 46ms)
- Inhibitory: α = 0.72 ± 0.03 (τ ≈ 35ms)

**Conditions:**
- Superseded by attention embedding (which implicitly captures dynamics)
- Combining with later improvements didn't help

---

## 8. Poisson Bregman Divergence (+16.1% from baseline)

**Evidence:**
- Phase 2 `poisson_ltrial`: 0.3369 vs baseline 0.2901

**Why It Works:**
Poisson divergence respects the natural variance structure of spike count data (variance scales with mean), providing a more appropriate distance metric for trial matching.

**Conditions:**
- Helped at lower baseline levels
- Hurt at higher baseline (Phase 5: 0.366 vs 0.402)
- Not recommended for high-performance configurations

---

## 9. ReduceLROnPlateau Scheduler

**Evidence:**
- All best configurations used plateau scheduler
- Cosine scheduler without warmup: 0.2801 vs baseline 0.2901

**Why It Works:**
Plateau scheduler reduces learning rate when validation loss stops improving, allowing fine-tuning in later training stages.

**Configuration:**
```python
scheduler: 'plateau'
lr_scheduler_patience: 50
lr_scheduler_factor: 0.5
min_lr: 1e-5
```

---

## Summary: Cumulative Improvement

| Step | Technique | Val Corr | Cumulative Gain |
|------|-----------|----------|-----------------|
| 0 | Baseline | 0.2901 | - |
| 1 | + Gradient balancing | 0.3299 | +13.7% |
| 2 | + L_trial scale 0.5 | 0.3614 | +24.6% |
| 3 | + Input embedding | 0.3762 | +29.7% |
| 4 | + Attention embedding | 0.3843 | +32.5% |
| 5 | + Learnable h0 | 0.4021 | +38.6% |
| 6 | + Extended training | 0.4121 | +42.1% |

**Key Insight:** Each major improvement addressed a different limitation:
1. Gradient balancing → loss optimization
2. Sinkhorn tuning → trial matching fidelity
3. Input embedding → input representation capacity
4. Attention → input feature interactions
5. Learnable h0 → initial condition matching
6. Extended training → convergence completion
