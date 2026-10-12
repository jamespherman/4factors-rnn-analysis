# Biological Input Experiments Report

**Date:** 2026-02-01
**Status:** Complete

---

## Executive Summary

This experiment tested the hypothesis that local SC computation—specifically E→E recurrence—is sufficient for encoding priority signals, while I→E inhibition mediates spatial competition rather than factor-specific competition.

### Key Findings

1. **Biological constraints preserve fit quality**: E1 biological achieved 0.3955 correlation (99% of baseline 0.3997)
2. **Strong emergent selectivity**: 89-100% of neurons not receiving direct input show significant selectivity for all factors
3. **E→E ablation preserves selectivity**: Removing E→E connections still allows 67-80% significant emergent selectivity
4. **I→E ablation breaks the model**: Removing I→E connections eliminates all activity (val_corr=0)

---

## Experimental Configuration

### Dataset
- **File**: `data/rnn_export_Newton_08_15_2025_SC.mat`
- **Neurons**: 42 (14 E, 28 I)
- **Trials**: 1337 (1070 train, 267 val)
- **Time bins**: 227 (25ms bins)

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Input embedding | Attention (56-dim, 4 heads) |
| Learnable h0 | Yes (init 0.1) |
| Gradient balancing | Yes (scale 0.5) |
| Max epochs | 2000 |
| Patience | 300 |
| Learning rate | 1e-3 → 1e-5 |

### Biological Input Fractions
| Factor | E fraction | I fraction | Rationale |
|--------|------------|------------|-----------|
| Reward | 30% | 40% | SNr pathway |
| Probability | 30% | 40% | SNr pathway |
| Salience | 75% | 25% | V1 pathway |
| Identity | 75% | 25% | V1 pathway |

---

## Results

### Experiment E0: Baseline (All neurons receive all inputs)

| Metric | Value |
|--------|-------|
| Best val correlation | 0.3997 |
| Best epoch | 1997 |
| Training time | 1:52:41 |
| Per-neuron correlation range | 0.04 - 0.91 |

**Note**: No emergent selectivity analysis possible since all neurons are "receivers" for all factors.

### Experiment E1: Biological Constraints

| Metric | Value |
|--------|-------|
| Best val correlation | 0.3955 |
| Best epoch | 907 |
| Training time | 0:35:02 |
| Per-neuron correlation range | 0.03 - 0.88 |

**Biological assignment statistics**:
| Factor | Total receivers | E receivers | I receivers |
|--------|-----------------|-------------|-------------|
| Reward | 15 (36%) | 4 | 11 |
| Probability | 15 (36%) | 4 | 11 |
| Salience | 17 (40%) | 10 | 7 |
| Identity | 17 (40%) | 10 | 7 |

### Emergent Selectivity in E1

| Factor | Mean selectivity | % Significant |
|--------|------------------|---------------|
| Reward | 0.384 | 100% |
| Probability | 0.194 | 89% |
| Salience | 0.363 | 100% |
| Identity | 0.325 | 100% |

**Interpretation**: Neurons that do NOT receive direct input for a factor nonetheless develop significant selectivity for that factor through recurrent computation.

---

## Ablation Analysis

### E1 Biological Model Ablations

| Ablation | Val Corr | Description |
|----------|----------|-------------|
| Full model | 0.3955 | All connections intact |
| full_recurrence | -0.007 | All W_rec = 0 |
| ie_only | 0.000 | I→E connections = 0 |
| ee_only | 0.012 | E→E connections = 0 |
| ei_only | 0.000 | E→I connections = 0 |

### Ablation Effects on Emergent Selectivity

| Ablation | Reward | Probability | Salience | Identity |
|----------|--------|-------------|----------|----------|
| Full model | 100% | 89% | 100% | 100% |
| full_recurrence | 59% | 70% | 80% | 72% |
| ie_only | 0% | 0% | 0% | 0% |
| ee_only | 67% | 78% | 80% | 80% |
| ei_only | 0% | 0% | 0% | 0% |

---

## Hypothesis Evaluation

### Original Hypothesis

> Local SC computation—specifically E→E recurrence—is sufficient for encoding priority signals. The architecture of local I→E inhibition is inconsistent with factor-specific competition: recorded interneurons display response properties consistent with mediating spatial (not factor) competition.

### Predictions and Results

| Prediction | Expected | Observed | Supported? |
|------------|----------|----------|------------|
| E→E propagates spatial factors | E→E ablation reduces reward/prob selectivity | E→E ablation: 67% reward, 78% prob significant | **Partial** |
| I→E suppresses spatial selectivity | I→E ablation increases reward/prob selectivity | I→E ablation: Model non-functional | **Untestable** |
| I→E enables feature selectivity | I→E ablation decreases salience/identity | I→E ablation: Model non-functional | **Untestable** |
| Emergent selectivity in non-receivers | 30%+ show significant selectivity | 89-100% significant | **Strongly supported** |

### Analysis

**Supported findings:**
1. ✅ Biological constraints achieve near-baseline fit (99%)
2. ✅ Strong emergent selectivity (89-100%) in neurons not receiving direct input
3. ✅ E→E ablation preserves significant emergent selectivity (67-80%)
4. ✅ Recurrence is required for model fit (full_recurrence ablation drops to -0.007)

**Confounded findings:**
1. ⚠️ I→E and E→I ablations completely disable the model (val_corr=0)
2. ⚠️ Cannot test differential effects of I→E on spatial vs feature factors
3. ⚠️ The 14E/28I ratio (inverted from typical cortical ratio) may affect results

**Interpretation:**
The model requires intact E-I communication for any function. With 28 I neurons (67% of recorded neurons), the I population is critical for network dynamics. Removing I→E or E→I connections eliminates all output.

However, the finding that E→E ablation (ee_only) preserves 67-80% significant emergent selectivity while reducing val_corr to 0.012 suggests that **multiple pathways contribute to emergent selectivity**, not just E→E recurrence.

---

## Conclusions

### Main Findings

1. **Biological input constraints are compatible with good model fits** (0.3955 vs 0.3997 baseline)

2. **Emergent selectivity is robust**: Neurons develop strong selectivity for factors they don't receive directly, demonstrating that recurrent computation transmits task-relevant information.

3. **E→E recurrence is not strictly necessary for emergent selectivity**: The ee_only ablation (removing E→E) still shows 67-80% significant emergent selectivity, suggesting other pathways (E→I→E or input embedding) contribute.

4. **E-I balance is critical**: Both I→E and E→I ablations completely disable the model, indicating that the balance between excitation and inhibition is essential for network function.

### Limitations

1. **Dataset composition**: 14E/28I is inverted from typical cortical E:I ratios (4:1), making comparisons to biological SC circuits difficult.

2. **Ablation sensitivity**: The complete failure of ie_only and ei_only ablations prevents testing differential effects on spatial vs feature factors.

3. **No direct comparison to original 54-neuron dataset**: The data was overwritten, preventing comparison to previous results.

### Future Directions

1. **Graded ablations**: Instead of complete removal, test partial ablations (e.g., 50% reduction in I→E weights) to see differential effects.

2. **Cell-type specific analysis**: Examine emergent selectivity separately for E and I neurons to test pathway-specific hypotheses.

3. **Larger dataset**: Use a dataset with more E neurons to test the hypothesis with typical E:I ratios.

---

## Files Generated

### Experiments
- `experiments/E0_baseline/results.json` - Full baseline results
- `experiments/E1_biological/results.json` - Full biological results
- `experiments/E0_baseline/model_best.pt` - Trained baseline model
- `experiments/E1_biological/model_best.pt` - Trained biological model

### Figures
- `figures/fit_quality_comparison.png` - E0 vs E1 correlation comparison
- `figures/emergent_selectivity_by_factor.png` - Selectivity by factor type
- `figures/emergent_selectivity_e_vs_i.png` - E vs I neuron selectivity
- `figures/ablation_by_factor.png` - Ablation effects on each factor

### Configuration
- `config.json` - Experiment configuration

---

*Report generated: 2026-02-01*
