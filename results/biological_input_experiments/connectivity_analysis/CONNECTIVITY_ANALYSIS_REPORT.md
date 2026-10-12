# Connectivity-Selectivity Analysis Report

**Date:** 2026-02-01

---

## Overview

This analysis examines the relationship between recurrent connectivity (E→E, I→E, etc.)
and factor selectivity in the trained RNN models. Selectivity is computed using ROC-AUC,
matching the neuronal analysis pipeline.

---

## Methods

### ROC-AUC Selectivity
- **Metric**: Area Under ROC Curve via Mann-Whitney U statistic
- **Range**: [0, 1], where 0.5 = chance
- **Significance**: Permutation test (n=200) + Bootstrap CI

### Visual Epoch Window
- **Start**: 50.0 ms post-target onset
- **End**: 250.0 ms post-target onset
- Matches SC visual epoch in neuronal analysis

---

## E0 Baseline Results

### ROC Selectivity Summary
- **Reward**: Mean AUC = 0.552, 14/42 significant
- **Probability**: Mean AUC = 0.456, 20/42 significant
- **Salience**: Mean AUC = 0.486, 18/42 significant
- **Identity**: Mean AUC = nan, 0/42 significant

### Weight-Selectivity Correlations (E0)

| Connection | Factor | Correlation | p-value | Significant |
|------------|--------|-------------|---------|-------------|
| EE | reward | 0.000 | 1.000 |  |
| EE | probability | 0.000 | 1.000 |  |
| EE | salience | 0.000 | 1.000 |  |
| EE | identity | 0.000 | 1.000 |  |
| EI | reward | 0.000 | 1.000 |  |
| EI | probability | 0.000 | 1.000 |  |
| EI | salience | 0.000 | 1.000 |  |
| EI | identity | 0.000 | 1.000 |  |
| IE | reward | 0.000 | 1.000 |  |
| IE | probability | 0.000 | 1.000 |  |
| IE | salience | 0.000 | 1.000 |  |
| IE | identity | 0.000 | 1.000 |  |
| II | reward | -0.023 | 0.520 |  |
| II | probability | -0.037 | 0.314 |  |
| II | salience | -0.048 | 0.183 |  |
| II | identity | 0.000 | 1.000 |  |

---

## Interpretation

### Weight-Selectivity Correlations
- **Positive correlation** in E→E: Neurons with similar selectivity have stronger excitatory connections
- **Negative correlation** in I→E: Inhibitory connections may suppress similar selectivity (competition)

### Effective Connectivity
- **β_E > 0**: Excitatory inputs from selective neurons increase target selectivity
- **β_I < 0**: Inhibitory inputs from selective neurons decrease target selectivity
- **β_direct**: Direct input contribution independent of recurrence

### Emergent Selectivity
Neurons that do NOT receive direct factor input but show significant selectivity
demonstrate that recurrent dynamics propagate factor information through the network.

---

## Generated Figures

1. `roc_selectivity_comparison.png` - E0 vs E1 selectivity scatter
2. `emergent_selectivity_roc.png` - Emergent selectivity boxplots
3. `weight_selectivity_scatter_*.png` - Weight vs selectivity by connection type
4. `effective_connectivity_coefficients.png` - β coefficients by factor
5. `connection_heatmap_*.png` - W_rec sorted by selectivity
6. `selectivity_flow_*.png` - Network visualization of selectivity propagation

---

*Report generated: {results['timestamp']}*
