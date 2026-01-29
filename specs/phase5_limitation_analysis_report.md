# Phase 5: Limitation Analysis Report

## Executive Summary

**New Best Validation Correlation: 0.4121** (extended_training, +2.5% vs Phase 4)

Phase 5 investigated what fundamentally limits model performance at the ~0.40 validation PSTH correlation level through systematic experiments and analytical deep-dives.

### Key Findings

1. **Training Duration Was Primary Bottleneck**: Extended training (1000 epochs) improved from 0.4021 to 0.4121 (+2.5%)
2. **No Single Architectural Change Helped**: All other modifications hurt performance
3. **Model Severely Under-Represents High Frequencies**: Power ratio drops to 0.002 at high frequencies
4. **Time Constants Not the Bottleneck**: Faster tau (25ms) hurt performance significantly

---

## Experiment Results Summary

| Experiment | Best Val Corr | vs Phase4 (0.402) | Conclusion |
|------------|---------------|-------------------|------------|
| **extended_training** | **0.4121** | **+0.010** | **NEW BEST** |
| low_rank_50 | 0.399 | -0.003 | Near baseline |
| noise_0.3 | 0.385 | -0.017 | Noise hurts |
| noise_0.5 | 0.384 | -0.018 | More noise hurts more |
| fast_tau_25 | 0.384 | -0.018 | Faster tau hurts |
| bypass_dale | 0.383 | -0.019 | Dale's law not the issue |
| learnable_tau | 0.388 | -0.014 | Learnable tau doesn't help |
| poisson_ltrial | 0.366 | -0.036 | Poisson hurts at high baseline |

---

## Detailed Experiment Analysis

### Experiment 0: Extended Training (1000 epochs)
**Result: best_val_corr = 0.4121 (+2.5%)**

- The Phase 4 model had not converged at 500 epochs
- Training to 1000 epochs yielded consistent improvement
- Final h0 values evolved significantly (mean: 0.51, range: -0.86 to 1.12)
- **Conclusion**: Model was still learning; extended training is beneficial

### Experiment 1: Faster Dynamics (tau=25ms)
**Result: best_val_corr = 0.384 (-4.5%)**

- Hypothesis: Model smoothness caused by slow time constants
- Result: Faster tau significantly hurt performance
- **Conclusion**: tau=50ms is appropriate; smoothness is NOT due to slow dynamics

### Experiment 2: Higher Noise (0.3 and 0.5)
**Results: 0.385 and 0.384 respectively**

- Hypothesis: Low Fano factor due to insufficient noise
- Result: Higher noise hurt correlation without helping variability
- **Conclusion**: Trial variability requires different approach than noise injection

### Experiment 3: Bypass Dale's Law
**Result: best_val_corr = 0.383 (-4.7%)**

- Hypothesis: E/I assignment constraints may be incorrect
- Result: Unconstrained weights performed worse
- **Conclusion**: Dale's law constraint is not limiting performance

### Experiment 4: Learnable Tau (Population)
**Result: best_val_corr = 0.388 (-3.5%)**

Learned values:
- tau_e = 47.7ms (slightly faster than 50ms init)
- tau_i = 37.7ms (slightly faster than 35ms init)
- alpha_e = 0.52, alpha_i = 0.66 (I neurons learned higher leak)

**Conclusion**: Learnable tau provides modest flexibility but doesn't improve overall fit

### Experiment 5: Poisson L_trial
**Result: best_val_corr = 0.366 (-8.9%)**

- Hypothesis: Poisson distance may help (previously helped at lower baseline)
- Result: Significantly hurt at high baseline
- **Conclusion**: Poisson L_trial interferes with high-performance training

### Experiment 6: Low-Rank Connectivity (rank=50)
**Result: best_val_corr = 0.399 (-0.8%)**

- Hypothesis: Regularization may improve generalization
- Result: Near-baseline performance with constrained connectivity
- **Conclusion**: Low-rank constraint is viable without much cost

---

## Analytical Deep-Dives

### I Neuron Analysis
From low_rank_50 model analysis:
- E neurons (n=41): mean correlation = 0.366, mean rate = 10.8 sp/s
- I neurons (n=13): mean correlation = 0.496, mean rate = 21.1 sp/s
- Top input driver for I neurons: loc4 (location 4)

**Note**: I neurons actually fit better in the low_rank model, contrary to Phase 4 findings. This suggests I neuron fit depends on model architecture.

### Spectral Analysis
Power spectral density comparison (model vs real):
- **Mean power ratio**: 0.099 (model has ~10% of real power)
- **Low frequency ratio**: 0.294 (model captures 29% at low frequencies)
- **High frequency ratio**: 0.002 (model severely under-represents high frequencies)

**Key Insight**: The model systematically loses high-frequency content, explaining the "too smooth" outputs identified in Phase 4 diagnostics.

---

## Conclusions

### Primary Limiting Factors

1. **Training Duration**: The model benefits from more training epochs (1000 vs 500)
2. **High-Frequency Dynamics**: Model architecture fundamentally smooths outputs
3. **Data Complexity**: ~40% correlation may represent a ceiling for this architecture

### What Didn't Help
- Faster time constants (tau=25ms)
- Higher noise injection (0.3, 0.5)
- Removing Dale's law constraints
- Poisson distance in L_trial
- Learnable per-population tau

### What Preserved Performance
- Low-rank connectivity (rank=50) - near baseline with regularization

---

## Recommendations

### Path Forward

Given that extended training is the only improvement and architectural changes don't help:

1. **For Model Improvement**:
   - Continue training the extended_training model longer (1500-2000 epochs)
   - Consider architectural changes to capture high-frequency dynamics (e.g., multi-timescale RNN)
   - Explore different loss functions that explicitly penalize smoothness

2. **For Scientific Analysis**:
   - The current 0.41 correlation model is suitable for analyzing learned connectivity patterns
   - Focus on E/I weight analysis, input selectivity, and attention patterns
   - The model captures the dominant temporal structure even if high-frequency details are smoothed

### Suggested Next Steps

1. Train final model to 2000 epochs to find true convergence
2. Analyze learned weights and attention patterns from best model
3. Compare learned connectivity to known SC circuit properties
4. Use model for hypothesis generation about SC computation

---

## Files Generated

### Results
- `results/phase5/extended_training_result.json` - Best model config/results
- `results/phase5/extended_training_model_best.pt` - Best model checkpoint
- `results/phase5/summary.json` - All experiment results

### Analysis Figures
- `results/phase5/analysis/experiment_comparison.png`
- `results/phase5/analysis/i_neuron_input_correlations.png`
- `results/phase5/analysis/autocorrelation_analysis.png`
- `results/phase5/analysis/signal_noise_analysis.png`
- `results/phase5/analysis/power_spectra_comparison.png`
- `results/phase5/analysis/ei_assignment_analysis.png`

---

## Summary Answers

1. **New best correlation**: 0.4121 (extended_training, 1000 epochs)
2. **Primary limiting factor**: Training duration; model smoothness at high frequencies
3. **Path forward**: Extended training helps; consider multi-timescale architectures or proceed with scientific analysis of current model
