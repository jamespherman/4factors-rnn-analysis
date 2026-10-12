# Untested Combinations: Promising Directions for Future Experiments

## Executive Summary

This document identifies **promising experiment combinations that have not been tried**. Based on what worked individually, we highlight potential synergies. However, the audit revealed a critical pattern: **combinations of individually successful techniques usually hurt performance**. Recommendations here are therefore **conservative** and focus on variations rather than stacking.

---

## 1. CAUTION: The Combination Problem

### Critical Finding from Audit
Every major combination tested in Phase 2c underperformed individual techniques:

| Combination | Result | Best Individual |
|-------------|--------|-----------------|
| alpha + poisson | 0.3083 | 0.3449 |
| alpha + low_rank | 0.2782 | 0.3449 |
| poisson + low_rank | 0.2929 | 0.3369 |
| h0 + AdamW | 0.3987 | 0.4021 |
| embed + alpha | 0.3167 | 0.3341 |

### Implication
Before testing combinations, consider:
1. Why each technique works (what limitation it addresses)
2. Whether they address the same or different limitations
3. Whether combining creates conflicting optimization pressures

---

## 2. Architecture Variations (Lower Risk)

### 2.1 Attention Head Counts
**Tested:** 4 heads
**Untested:** 1, 2, 8, 16 heads

| Heads | Expected Effect | Priority |
|-------|-----------------|----------|
| 1 | May reduce overfitting | Medium |
| 2 | Balanced complexity | Medium |
| 8 | More interaction modeling | Low |
| 16 | Likely overfitting | Low |

**Recommendation:** Test 2 heads as potentially simpler alternative.

### 2.2 Embedding Dimension with Attention
**Tested:** 56 with attention
**Untested:** 28, 42, 70, 84 with attention

Current 4x expansion may not be optimal for attention (designed for linear).

**Recommendation:** Test 42 (3x) with attention for potentially better regularization.

### 2.3 Multi-Head Attention Architecture
**Tested:** Single attention layer
**Untested:**
- Multi-layer attention (2-3 layers)
- Attention with residual connections
- Cross-attention between input streams

**Risk:** Higher complexity, may overfit

---

## 3. Optimization Variations (Lower Risk)

### 3.1 Learning Rate
**Tested:** 0.001, 0.0005
**Untested:** 0.003, 0.01

Higher LR may accelerate convergence if stable.

### 3.2 Scheduler Combinations
**Tested:** Plateau alone, Cosine alone
**Untested:**
- Plateau + Warmup
- OneCycleLR
- Exponential decay

### 3.3 Gradient Clipping Values
**Tested:** 1.0 only
**Untested:** 0.5, 2.0, 5.0

May affect training stability and final performance.

---

## 4. Training Duration Experiments

### 4.1 Very Extended Training
**Tested:** 500, 1000, 2000 epochs
**Untested:** 5000, 10000 epochs

Phase 5 showed +2.5% from 500→1000. More may help.

### 4.2 Patience Variations
**Tested:** 100, 150, 200, 300
**Untested:** Patience = max_epochs (no early stopping)

Compare final vs early-stopped performance.

---

## 5. Loss Function Variations (Medium Risk)

### 5.1 L_trial Scale
**Tested:** 0.25, 0.5, 1.0, 2.0
**Untested:** 0.1, 0.75

Finer granularity around optimal (0.5).

### 5.2 Sinkhorn Epsilon
**Tested:** 0.05, 0.1, 0.2, 0.5, 1.0
**Untested:** 0.01, 0.03, 0.15

Very tight epsilon (0.01-0.03) might improve trial matching.

### 5.3 Alternative Trial Matching
**Tested:** Sinkhorn optimal transport
**Untested:**
- Hungarian algorithm (exact matching)
- Chamfer distance
- Earth mover's distance variants

---

## 6. Regularization Experiments

### 6.1 Dropout
**Tested:** None
**Untested:** Dropout on embeddings, recurrent connections

May help with generalization.

### 6.2 Weight Decay Values
**Tested:** 0.01 (AdamW)
**Untested:** 0.001, 0.1

### 6.3 Spectral Normalization
**Tested:** None
**Untested:** Spectral normalization on W_rec

May improve training stability.

---

## 7. Data Variations (Medium Risk)

### 7.1 Data Augmentation
**Tested:** None
**Untested:**
- Time jittering
- Trial shuffling
- Noise injection on targets

### 7.2 Train/Val Split Variations
**Tested:** 80/20 fixed
**Untested:**
- 70/30
- 90/10
- Cross-validation (5-fold)

### 7.3 Condition-Stratified Training
**Tested:** Uniform trial weighting
**Untested:** Upweight rare conditions

---

## 8. Higher-Risk Combinations

These combine techniques that individually helped but may conflict:

### 8.1 Attention + Learnable Alpha (Tested, Hurt)
Already tested: 0.3789 vs 0.3907 baseline
**Don't retry** - confirmed interference.

### 8.2 h0 + Low-Rank (NOT Tested)
| Component | Individual Effect |
|-----------|------------------|
| learnable_h0 | +4.6% |
| low_rank_50 | -0.8% at high baseline |

**Risk:** Medium - low_rank provides regularization
**Recommendation:** Low priority, but worth one test.

### 8.3 Extended Training + Low-Rank (NOT Tested)
May benefit from regularization during long training.

**Recommendation:** Worth testing if overfitting becomes issue.

### 8.4 Attention + Recurrent Embedding (NOT Tested)
GRU preprocessing before attention.

**Risk:** High complexity
**Recommendation:** Low priority.

---

## 9. Novel Directions (Higher Risk)

### 9.1 Multi-Timescale RNN
**Motivation:** Phase 5 showed model under-represents high frequencies.

**Design:**
- Fast pathway (tau=25ms) + Slow pathway (tau=100ms)
- Combined output

**Risk:** Significant architecture change.

### 9.2 Separate Loss Weighting per Neuron Type
**Motivation:** I neurons fit better than E neurons consistently.

**Design:**
- Higher loss weight for E neurons
- Learn per-neuron loss weights

### 9.3 PCA-Space Loss
**Motivation:** Phase 4 showed PC2/PC3 correlation much lower than PC1.

**Design:**
- Add loss term penalizing PCA projection error
- May improve population-level dynamics

### 9.4 Fano Factor Matching
**Motivation:** Model Fano factor is 17x lower than data.

**Design:**
- Add explicit loss term for variability matching
- May conflict with PSTH matching

---

## 10. Cross-Session Experiments

### 10.1 Pooled Training
**Tested:** Each session separately
**Untested:**
- Pool all Newton sessions
- Session as covariate

### 10.2 Transfer Learning
**Tested:** None
**Untested:**
- Pretrain on largest session (08_15)
- Fine-tune on smaller sessions (08_13, 08_14)

### 10.3 Multi-Animal
**Tested:** Newton only
**Untested:** Feynman sessions (if available)

---

## Priority Ranking

### High Priority (Low Risk, Potential Benefit)
1. **Attention with 2 heads** - simpler may generalize better
2. **Very tight Sinkhorn (epsilon=0.01)** - may improve trial matching
3. **L_trial scale = 0.75** - finer exploration around optimal
4. **Extended training to 5000 epochs** - continued learning

### Medium Priority
5. **Dropout on embeddings** - regularization
6. **PCA-space loss term** - population structure
7. **Pooled training across sessions** - more data

### Low Priority (Higher Risk)
8. h0 + low_rank combination
9. Multi-timescale architecture
10. Per-neuron loss weighting

---

## What NOT to Try

Based on consistent failures:

1. **Any triple combination** - All tested triples hurt badly
2. **Poisson L_trial with high baseline** - Confirmed interference
3. **Learnable alpha with attention** - Tested, hurt
4. **h0 + AdamW** - Tested, slight hurt
5. **Large embeddings (>56 dim)** - Overfitting confirmed
6. **Soft Sinkhorn (epsilon >= 0.5)** - Consistent hurt
7. **Fixed lambda weighting** - Always worse than gradient balancing
8. **Fast tau (25ms)** - Tested, hurt
9. **High noise (>0.1)** - Tested, hurt
10. **Bypassing Dale's law** - Tested, hurt
