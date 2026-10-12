# What Didn't Work: Techniques That Hurt or Had No Effect

## Executive Summary

Many promising techniques either hurt performance or showed no benefit when combined with the best configuration. The most important finding is that **combining individually successful techniques usually hurts performance**. This document catalogs failures to prevent re-testing and inform future directions.

---

## 1. Combining Multiple Improvements (CONSISTENTLY HURT)

**Evidence:**
| Combination | Result | Best Individual |
|-------------|--------|-----------------|
| alpha_neuron + poisson_ltrial | 0.3083 | 0.3449 (alpha) |
| alpha_neuron + low_rank_50 | 0.2782 | 0.3449 (alpha) |
| poisson_ltrial + low_rank_50 | 0.2929 | 0.3369 (poisson) |
| alpha + poisson + low_rank | 0.2746 | **WORST** |
| alpha + poisson + activity_reg | 0.3105 | 0.3449 (alpha) |
| embed_28 + alpha_neuron | 0.3167 | 0.3341 (embed) |
| attention + h0 + AdamW | 0.3987 | 0.4021 (h0 alone) |

**Why It Failed:**
- Improvements address similar underlying limitations (expressiveness vs regularization)
- Multiple changes create conflicting optimization pressures
- Each improvement was tuned for the baseline, not for combination
- Gradient balancing may not handle multiple loss modifications well

**Recommendation:**
Do NOT attempt to combine techniques that each individually improved performance. The best configuration uses specific techniques that complement each other (attention + h0 + gradient balancing) rather than stacking all positive findings.

---

## 2. Fixed Lambda Weighting for L_trial (ALL BELOW BASELINE)

**Evidence:**
| Config | Lambda | Val Corr | vs Baseline |
|--------|--------|----------|-------------|
| baseline_no_ltrial | 0 | 0.3299 | - |
| lambda_0.1_eps_0.1 | 0.1 | 0.3185 | -3.5% |
| lambda_0.3_eps_0.1 | 0.3 | 0.1877 | -43.1% |
| lambda_0.5_eps_0.1 | 0.5 | 0.1731 | -47.5% |
| lambda_1.0_eps_0.1 | 1.0 | 0.1714 | -48.0% |

**Why It Failed:**
Fixed lambda creates gradient conflicts where L_trial dominates optimization when its gradients are much larger than L_neuron's. This prevents proper PSTH matching. Higher lambda values are catastrophic.

**Recommendation:**
Always use gradient balancing instead of fixed lambda weighting. Never use lambda >= 0.3.

---

## 3. Soft Sinkhorn Epsilon (HURT PERFORMANCE)

**Evidence:**
| Epsilon | With GB | Without GB |
|---------|---------|------------|
| 0.05 | 0.3559 | - |
| 0.1 | 0.3434 | 0.3185 |
| 0.5 | 0.3040 | 0.1767 |
| 1.0 | 0.3202 | 0.2281 |

**Why It Failed:**
Higher epsilon produces softer (more entropic) assignment matrices in optimal transport, losing the bijective matching property needed for proper trial alignment. The assignments become too diffuse to capture trial-specific structure.

**Recommendation:**
Keep epsilon at 0.05-0.1. Never use epsilon >= 0.5.

---

## 4. L_trial Warmup (HURT PERFORMANCE)

**Evidence:**
| Config | Warmup Epochs | Val Corr |
|--------|---------------|----------|
| No warmup | 0 | 0.3434 |
| warmup_50 | 50 | 0.2949 |
| warmup_100 | 100 | 0.3168 |

**Why It Failed:**
Delaying L_trial introduction prevented early co-optimization of PSTH and trial-matching objectives. The model converged to a solution that fits PSTHs but not single-trial dynamics.

**Recommendation:**
Include L_trial from epoch 0 with gradient balancing.

---

## 5. Larger Input Embeddings (8x, 12x) (HURT PERFORMANCE)

**Evidence:**
| Embedding Dim | Expansion | Val Corr | vs 56-dim |
|---------------|-----------|----------|-----------|
| 56 | 4x | 0.3762 | baseline |
| 112 | 8x | 0.3440 | -8.6% |
| 168 | 12x | 0.3120 | -17.1% |

**Why It Failed:**
Larger embeddings likely overfit or make optimization harder. More parameters require more data or stronger regularization.

**Recommendation:**
Stick with 4x expansion (56 dimensions for 14 inputs).

---

## 6. Learnable Tau (Original Implementation) (CATASTROPHIC FAILURE)

**Evidence:**
| Config | Val Corr | Failure Mode |
|--------|----------|--------------|
| learnable_tau_population | 0.0205 | alpha > 1 |
| learnable_tau_neuron | 0.0234 | alpha > 1 |

**Why It Failed:**
Original TAU_MIN = 10ms was less than dt = 25ms. When tau dropped below dt, the integration constant alpha = dt/tau exceeded 1, breaking the exponential moving average dynamics:
```
x = (1 - alpha) * x + alpha * input  # Invalid when alpha > 1
```

**Fix Applied:**
Phase 2c fixed this by setting TAU_MIN >= dt (25ms).

**After Fix:**
`learnable_tau_pop_v2`: 0.3461 (works, but doesn't beat learnable_alpha)

---

## 7. Faster Time Constants (tau=25ms) (HURT PERFORMANCE)

**Evidence:**
Phase 5 `fast_tau_25`: 0.384 vs best: 0.402 (-4.5%)

**Why It Failed:**
The hypothesis was that model smoothness was due to slow time constants. However, tau=50ms is appropriate for SC dynamics. Faster tau made outputs too noisy without improving high-frequency content.

**Recommendation:**
Keep tau=50ms as default.

---

## 8. Higher Noise Injection (0.3, 0.5) (HURT PERFORMANCE)

**Evidence:**
| Noise Scale | Val Corr |
|-------------|----------|
| 0.1 (default) | 0.402 |
| 0.3 | 0.385 |
| 0.5 | 0.384 |

**Why It Failed:**
The hypothesis was that low Fano factor was due to insufficient noise. However, higher noise hurt correlation without improving trial-to-trial variability matching.

**Insight:**
Trial variability requires a different approach than noise injection. The model architecture may fundamentally limit expressible variability.

---

## 9. Bypassing Dale's Law (HURT PERFORMANCE)

**Evidence:**
Phase 5 `bypass_dale`: 0.383 vs best: 0.402 (-4.7%)

**Why It Failed:**
The hypothesis was that E/I assignment constraints might be incorrect. However, unconstrained weights performed worse, suggesting:
1. Dale's law provides useful regularization
2. The E/I classification from data is appropriate

**Recommendation:**
Keep Dale's law enforcement.

---

## 10. Typed/Separate Embeddings (HURT PERFORMANCE)

**Evidence:**
Phase 3 `typed_embedding`: 0.3305 vs `attention_embedding`: 0.3843 (-14%)

**Why It Failed:**
Separate embeddings per input group (location, identity, reward) reduced cross-input interactions that attention captures effectively.

---

## 11. Activity Regularization (λ=0.01) (TOO STRONG)

**Evidence:**
| Lambda | Val Corr |
|--------|----------|
| 0.001 | 0.3171 (+9.3%) |
| 0.01 | 0.2828 (-2.5%) |

**Why It Failed:**
Strong activity regularization suppressed neural activity too much, hurting PSTH matching.

---

## 12. Cosine Scheduler Without Warmup (HURT PERFORMANCE)

**Evidence:**
`cosine_no_warmup`: 0.2801 vs baseline 0.2901 (-3.4%)

**Why It Failed:**
Cosine annealing reduced learning rate too quickly without sufficient initial training.

---

## 13. Poisson L_trial at High Baseline (HURT PERFORMANCE)

**Evidence:**
Phase 5 `poisson_ltrial`: 0.366 vs best: 0.402 (-8.9%)

**Why It Failed:**
Poisson distance helped at lower baseline levels (Phase 2: 0.337 vs 0.290) but interfered with high-performance training. May conflict with gradient balancing or attention-based representations.

**Recommendation:**
Don't use Poisson L_trial with attention + h0 configuration.

---

## 14. Per-Neuron Learnable Alpha with Attention (HURT PERFORMANCE)

**Evidence:**
Phase 4 `attention_alpha_neuron`: 0.3789 vs `attention_baseline`: 0.3907 (-3%)

**Why It Failed:**
Attention embedding may implicitly handle temporal dynamics that learnable alpha was meant to capture. Adding learnable alpha introduces redundant parameters.

---

## Summary: Avoid These Patterns

1. **Don't stack improvements** - Each helps alone but combinations hurt
2. **Don't use fixed lambda weighting** - Use gradient balancing
3. **Don't use soft Sinkhorn (epsilon >= 0.5)** - Keep tight (0.05-0.1)
4. **Don't use warmup for L_trial** - Include from epoch 0
5. **Don't use very large embeddings** - 4x expansion is optimal
6. **Don't bypass biological constraints** - Dale's law helps
7. **Don't add learnable dynamics with attention** - Redundant complexity
8. **Don't use high noise** - 0.1 is optimal
9. **Don't use faster tau** - 50ms is appropriate

---

## Worth Revisiting?

| Technique | Verdict | Condition to Revisit |
|-----------|---------|---------------------|
| Combinations | No | Fundamental conflict |
| Fixed lambda | No | Gradient balancing superior |
| Soft Sinkhorn | No | Tight epsilon required |
| Large embeddings | Maybe | With stronger regularization |
| Learnable alpha | Maybe | Different architecture |
| High noise | Maybe | Different loss function |
| Poisson L_trial | Maybe | Lower baseline starting point |
