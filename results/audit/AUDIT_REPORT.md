# Experiment Audit Report

**Date:** 2026-01-31
**Repository:** 4factors-rnn-analysis
**Scope:** Comprehensive audit of all RNN fitting experiments

---

## Executive Summary

This audit catalogued **98 experiments** conducted between January 22-30, 2026, testing various modifications to the E-I constrained RNN for fitting neural recordings from primate superior colliculus. The experiments systematically improved validation PSTH correlation from a **baseline of 0.2901 to a best of 0.4121** (+42.1% improvement).

### Key Numbers

| Metric | Value |
|--------|-------|
| Total experiments catalogued | 98 |
| Best validation correlation | **0.4121** (extended_training) |
| Production model correlation | **0.4070** (final_model) |
| Baseline correlation | 0.2901 |
| Total improvement | +42.1% |
| Failed experiments | 2 (learnable tau bug, fixed) |

---

## Best Configuration (Production)

```python
{
    "learning_rate": 0.001,
    "max_epochs": 2000,
    "patience": 300,
    "tau": 50.0,
    "noise_scale": 0.1,
    "input_embed_dim": 56,
    "input_embed_type": "attention",
    "attention_heads": 4,
    "learnable_h0": true,
    "h0_init": 0.1,
    "use_grad_balancing": true,
    "ltrial_scale": 0.5,
    "sinkhorn_epsilon": 0.1,
    "scheduler": "plateau"
}
```

---

## Top 5 Key Findings

### 1. Attention-Based Input Embedding is Transformative (+32.5%)
Self-attention over input channels was the single biggest architectural improvement. It outperformed linear embeddings, recurrent embeddings, and all other variants.

**Evidence:** Phase 3 attention_embedding: 0.3843 vs baseline 0.2901

### 2. Learnable Initial State (h0) Achieves Target (+4.6%)
Learning a non-zero initial hidden state allowed the network to match non-zero firing rates at trial onset. This was the modification that crossed the 0.40 threshold.

**Evidence:** Phase 4 attention_learnable_h0: 0.4021 vs attention_baseline: 0.3907

### 3. Combining Improvements Usually Hurts
Every major combination tested underperformed individual techniques. This is the most surprising and important finding - it means "stacking" successful techniques is not a viable strategy.

**Evidence:** All Phase 2c combinations (alpha+poisson, alpha+lowrank, etc.) performed worse than individual components.

### 4. Gradient Balancing is Essential for L_trial (+9.5%)
Fixed lambda weighting for trial-matching loss consistently hurt performance. Gradient balancing, which normalizes gradient magnitudes, is required.

**Evidence:** gradbal_scale_0.5: 0.3614 vs baseline_no_ltrial: 0.3299

### 5. Extended Training Helps (+2.5%)
The model was still learning at 500 epochs. Training to 1000-2000 epochs provided continued improvement.

**Evidence:** extended_training (1000 epochs): 0.4121 vs attention_learnable_h0 (500 epochs): 0.4021

---

## What Definitively Helps

| Technique | Effect Size | Confidence |
|-----------|-------------|------------|
| Attention embedding | +32.5% | HIGH |
| Input embedding (56 dim) | +29.7% | HIGH |
| Learnable h0 | +4.6% | HIGH |
| Gradient balancing | +9.5% | HIGH |
| L_trial scale 0.5 | +5% rel | HIGH |
| Tight Sinkhorn (eps=0.1) | +5% rel | HIGH |
| Extended training | +2.5% | HIGH |
| ReduceLROnPlateau | best scheduler | MEDIUM |

---

## What Definitively Hurts

| Technique | Effect Size | Confidence |
|-----------|-------------|------------|
| Fixed lambda for L_trial | -15% to -48% | HIGH |
| Soft Sinkhorn (eps>=0.5) | -10% | HIGH |
| Large embeddings (>56) | -8% to -17% | HIGH |
| Combining improvements | -5% to -20% | HIGH |
| L_trial warmup | -8% | HIGH |
| High noise (>0.1) | -4% | HIGH |
| Faster tau (25ms) | -4% | MEDIUM |
| Bypassing Dale's law | -5% | MEDIUM |
| Learnable alpha w/ attention | -3% | MEDIUM |

---

## Open Questions

1. **Why do combinations hurt?**
   - Each technique may address the same underlying limitation differently
   - Conflicting optimization pressures between modifications
   - Techniques tuned for baseline, not for each other

2. **What limits performance at ~0.41?**
   - Model under-represents high-frequency content (power ratio 0.002)
   - Fano factor mismatch (model 17x lower than data)
   - May be architectural limitation

3. **Do findings generalize across animals?**
   - Only Newton sessions tested
   - Feynman data not explored
   - Cross-session performance varies (0.32-0.41)

4. **Is I neuron fitting better because of the model or the data?**
   - I neurons consistently fit better across all sessions
   - May reflect I neuron response properties

---

## Recommended Next Experiments

### High Priority (Low Risk)

1. **Test 2-head attention** (vs 4-head)
   - Simpler may generalize better
   - Untested variation

2. **Very tight Sinkhorn (epsilon=0.01-0.03)**
   - Current best is 0.05-0.1
   - May improve trial matching

3. **Even longer training (5000 epochs)**
   - Model still learning at 2000
   - Check for continued improvement

### Medium Priority

4. **PCA-space loss term**
   - PC2/PC3 correlation much lower than PC1
   - May improve population dynamics

5. **Dropout on attention embeddings**
   - Regularization not fully explored
   - May help generalization

6. **Cross-session pooled training**
   - More data may improve learning
   - Session as covariate

### Lower Priority

7. **Multi-timescale RNN**
   - Address high-frequency limitation
   - Significant architecture change

8. **Per-neuron loss weighting**
   - Improve E neuron fitting
   - May conflict with I neurons

---

## Files Generated

```
results/audit/
├── AUDIT_REPORT.md          # This file
├── experiments_table.csv     # Master table (98 experiments)
├── experiments_table_summary.md  # Readable summary
├── what_worked.md           # Successful techniques
├── what_didnt_work.md       # Failed techniques
├── optimization_findings.md  # Training parameters
├── architecture_findings.md  # Model structure
├── input_design_findings.md  # Input structure
├── untested_combinations.md  # Future directions
├── generate_figures.py      # Visualization script
└── figures/
    ├── fit_quality_over_time.png
    ├── parameter_vs_performance.png
    ├── configuration_heatmap.png
    └── top_experiments_comparison.png
```

---

## Improvement Timeline

```
Phase       Best Val Corr   Improvement   Key Technique
────────────────────────────────────────────────────────
Baseline    0.2901          -             Reference
L_trial     0.3614          +24.6%        Gradient balancing
Phase 2     0.3449          +18.9%        Learnable alpha
Phase 2c    0.3762          +29.7%        Input embedding 56
Phase 3     0.3843          +32.5%        Attention embedding
Phase 4     0.4021          +38.6%        Learnable h0 (TARGET)
Phase 5     0.4121          +42.1%        Extended training
Final       0.4070          +40.3%        Production (2000 epochs)
```

---

## Cross-Session Replication

| Session | Neurons | Val Corr | Key Findings |
|---------|---------|----------|--------------|
| 08_15 | 54 (41E/13I) | 0.4070 | Primary session |
| 08_14 | 80 (25E/55I) | 0.3608 | More I neurons |
| 08_13 | 20 (10E/10I) | 0.3232 | Small session |

**Replicated findings:**
- Global inhibition pattern (I neurons project broadly)
- I neurons have higher h0 than E neurons
- Model fits I neurons better than E neurons

**Partially replicated:**
- Effective target percentage (varies 36-69%)
- h0 E/I difference significance (1/3 significant)

---

## Conclusion

This audit provides a comprehensive map of the experimental landscape. The path from baseline (0.29) to best (0.41) involved:

1. **Better loss optimization** (gradient balancing, Sinkhorn tuning)
2. **Richer input representations** (embedding, attention)
3. **Better initial conditions** (learnable h0)
4. **Sufficient training** (1000-2000 epochs)

The most important lesson is that **combining individually successful techniques usually hurts**. The best configuration is not a stack of all positive findings, but a carefully selected set of complementary modifications.

Future work should focus on:
- Addressing the high-frequency limitation
- Exploring multi-animal generalization
- Testing fine-grained variations of successful techniques

---

*Report generated by comprehensive experiment audit, January 31, 2026*
