# Connectivity-Selectivity Analysis Report (v2)

**Date:** 2026-02-01

---

## Overview

This analysis examines the relationship between recurrent connectivity (E→E, I→E, etc.) and factor selectivity using ROC-AUC, matching the neuronal analysis pipeline from `4factors-analysis-pipeline`.

### Data Configuration
- **Recorded E neurons:** 14
- **Recorded I neurons:** 28
- **Total recorded neurons:** 42
- **Model E neurons:** 112 (includes 98 hidden E neurons)
- **Model I neurons:** 28
- **Model total neurons:** 140
- **Visual epoch:** bins 41-49 (50-250ms post-target onset)

---

## Detailed Methodology

### 1. ROC-AUC Selectivity Computation

**Implementation:** `src/analysis/roc_selectivity.py`

Selectivity is computed using ROC-AUC via the Mann-Whitney U statistic, matching the approach in `4factors-analysis-pipeline/code/utils/arrayROC.m`.

**Formula:**
```
AUC = U₂ / (n₁ × n₂)

where:
- U₂ = (sum of ranks for condition 2) - n₂(n₂+1)/2
- n₁ = trials in condition 1 (e.g., low reward)
- n₂ = trials in condition 2 (e.g., high reward)
```

**Interpretation:**
- AUC = 0.5: No selectivity (chance)
- AUC > 0.5: Higher firing for condition 2
- AUC < 0.5: Higher firing for condition 1

**Code reference:** `compute_roc_auc()` in `src/analysis/roc_selectivity.py:69-105`

### 2. Statistical Testing

**Permutation Test (p-value):**
- Shuffle condition labels `n_perm=200` times
- Compute null AUC distribution
- Two-sided p-value: fraction of null |AUC - 0.5| ≥ observed |AUC - 0.5|

**Bootstrap Confidence Interval:**
- Resample each condition independently `n_bootstrap=200` times
- Compute 95% CI from bootstrap AUC distribution

**Significance criterion:** p < 0.05 AND CI excludes 0.5

**Code reference:** `permutation_test_roc()` and `bootstrap_ci_roc()` in `src/analysis/roc_selectivity.py:108-185`

### 3. Visual Epoch Window

**Parameters:**
- Window: 50-250ms post-target onset (matches SC visual epoch in neuronal pipeline)
- Bin size: 25ms
- Start bin: target_onset_bin + 2 = bin 41
- End bin: target_onset_bin + 10 = bin 49

Firing rates are averaged within this window before ROC computation.

**Code reference:** `find_visual_epoch_bins()` in `src/analysis/roc_selectivity.py:188-214`

### 4. Weight-Selectivity Correlation Analysis

**Method:** Pearson correlation between |weight| and selectivity similarity

For each connection pair (i, j):
- Weight: |W_rec[i, j]|
- Selectivity similarity: -|AUC_i - AUC_j| (negative distance so positive r = similar selectivity → stronger connection)

**Connection types analyzed:**
- E→E: Excitatory to excitatory
- I→E: Inhibitory to excitatory
- E→I: Excitatory to inhibitory
- I→I: Inhibitory to inhibitory

**Note on neuron indexing:**
The model has 140 neurons (112 E + 28 I) but only 42 are recorded (14 E + 28 I).
Weight blocks are extracted for recorded neurons only:
```python
# Recorded E: indices 0:14
# Hidden E: indices 14:112
# Recorded I: indices 112:140

'EE': W_rec[:14, :14]           # E→E for recorded
'IE': W_rec[:14, 112:140]       # I→E for recorded
'EI': W_rec[112:140, :14]       # E→I for recorded
'II': W_rec[112:140, 112:140]   # I→I for recorded
```

**Code reference:** `extract_recorded_weight_blocks()` and `analyze_weight_selectivity_correlation_recorded()` in `scripts/analyze_connectivity_selectivity_v2.py:74-172`

### 5. Effective Connectivity Model

**Regression model for E neurons:**
```
selectivity[i] = β₀ + β_direct × receives[i] + β_EE × E_input[i] + β_IE × I_input[i]
```

Where:
- `receives[i]`: Binary indicator if neuron i receives direct factor input
- `E_input[i]`: Σⱼ |W_EE[i,j]| × selectivity[j] (weighted E input)
- `I_input[i]`: Σⱼ |W_IE[i,j]| × selectivity[j] (weighted I input)

**Interpretation:**
- β_direct > 0: Direct input increases selectivity
- β_EE > 0: E input from selective neurons increases selectivity
- β_IE < 0: I input from selective neurons decreases selectivity

**Code reference:** `fit_effective_connectivity_model_recorded()` in `scripts/analyze_connectivity_selectivity_v2.py:200-284`

### 6. Emergent Selectivity Analysis

**Definition:** Neurons that do NOT receive direct factor input but show significant selectivity.

For E1 (biological input assignment):
- Some neurons receive direct factor input (determined by `InputAssignment`)
- "Non-receivers" show selectivity through recurrent dynamics alone
- This demonstrates information propagation through network connectivity

**Code reference:** `compute_emergent_roc_selectivity()` in `src/analysis/roc_selectivity.py:404-453`

### 7. Graded Ablation Analysis

**Method:** Scale connection types from 0 (ablated) to 1 (intact)

Scale factors: [0.0, 0.25, 0.5, 0.75, 1.0]

For each scale factor:
1. Multiply weight block by scale
2. Run model forward pass
3. Compute mean |AUC - 0.5| (selectivity strength)

This reveals which connection types are most critical for selectivity.

**Code reference:** `graded_ablation_analysis()` in `scripts/analyze_connectivity_selectivity_v2.py:287-364`

---

## E0 Baseline Results

### ROC Selectivity
- **Reward:** Mean AUC = 0.535, 8/42 significant
- **Probability:** Mean AUC = 0.463, 15/42 significant
- **Salience:** Mean AUC = 0.497, 19/42 significant
- **Identity:** Mean AUC = nan, 0/42 significant (excluded - only one unique value in validation set)

### Weight-Selectivity Correlations

| Factor | E→E | I→E | E→I | I→I |
|--------|-----|-----|-----|-----|
| Reward | 0.179* | 0.042 | -0.024 | -0.042 |
| Probability | 0.167* | 0.023 | -0.006 | -0.047 |
| Salience | 0.153* | 0.014 | -0.045 | -0.045 |
| Identity | 0.000 | 0.000 | 0.000 | 0.000 |

\* p < 0.05

**Interpretation:** Positive E→E correlations indicate that neurons with similar selectivity have stronger excitatory connections to each other.

---

## E1 Biological Results

### ROC Selectivity & Emergent Selectivity
- **Reward:** Mean AUC = 0.775, **96% emergent significant**
- **Probability:** Mean AUC = 0.323, **93% emergent significant**
- **Salience:** Mean AUC = 0.305, **88% emergent significant**
- **Identity:** Mean AUC = nan, **0% emergent significant**

### Weight-Selectivity Correlations

| Factor | E→E | I→E | E→I | I→I |
|--------|-----|-----|-----|-----|
| Reward | -0.021 | 0.019 | -0.091 | -0.047 |
| Probability | -0.056 | 0.021 | -0.113* | -0.058 |
| Salience | -0.084 | 0.027 | 0.049 | -0.050 |
| Identity | 0.000 | 0.000 | 0.000 | 0.000 |

### Effective Connectivity Model

**E Neurons:** `selectivity = β₀ + β_direct×receives + β_EE×E_input + β_IE×I_input`

| Factor | β_direct | β_EE | β_IE | R² |
|--------|----------|------|------|----|
| Reward | -0.1110 | -1.5623 | 0.6368 | 0.241 |
| Probability | 0.1582 | 0.0544 | 0.1888 | 0.101 |
| Salience | 0.1952 | -0.0517 | -0.1057 | 0.239 |

---

## Key Findings

### 1. ROC-Based Selectivity Matches Neuronal Pipeline
Selectivity computed using ROC-AUC (Mann-Whitney U) in the 50-250ms visual epoch, directly comparable to neuronal selectivity measures from `4factors-analysis-pipeline`.

### 2. Strong Emergent Selectivity in Biological Model (E1)
Neurons NOT receiving direct factor input show significant selectivity:
- **Reward:** 96% of non-receivers significant
- **Probability:** 93% of non-receivers significant
- **Salience:** 88% of non-receivers significant
- **Identity:** 0% of non-receivers significant

This demonstrates that recurrent dynamics propagate factor information through the network.

### 3. E→E Weight-Selectivity Correlations (E0)
In the baseline model (E0), E→E connections show significant positive correlations with selectivity similarity (r ≈ 0.15-0.18, p < 0.05). This pattern is not present in E1.

### 4. Effective Connectivity Explains ~10-24% of Variance
The regression model captures meaningful structure, with reward and salience showing R² ≈ 0.24.

---

## File Locations

### Source Code
- **Main analysis script:** `scripts/analyze_connectivity_selectivity_v2.py`
- **ROC selectivity module:** `src/analysis/roc_selectivity.py`
- **Input assignment:** `src/input_assignment.py`

### Model Checkpoints
- **E0 Baseline:** `results/biological_input_experiments/experiments/E0_baseline/model_best.pt`
- **E1 Biological:** `results/biological_input_experiments/experiments/E1_biological/model_best.pt`

### Output Files
- **Full results:** `results/biological_input_experiments/connectivity_analysis/results_v2.json`
- **This report:** `results/biological_input_experiments/connectivity_analysis/CONNECTIVITY_ANALYSIS_REPORT_v2.md`

### Generated Figures
Located in `results/biological_input_experiments/connectivity_analysis/figures/`:

1. `roc_selectivity_e0.png` / `roc_selectivity_e1.png` - ROC selectivity distributions by neuron type
2. `weight_sel_scatter_e0_*.png` / `weight_sel_scatter_e1_*.png` - Weight vs selectivity similarity (4 factors × 2 experiments)
3. `effective_connectivity_e1.png` - Regression β coefficients
4. `graded_ablation_e1.png` - Selectivity vs connection scale factor
5. `emergent_selectivity_e1.png` - Receivers vs non-receivers comparison

---

## How to Reproduce

```bash
# Run the connectivity-selectivity analysis
python scripts/analyze_connectivity_selectivity_v2.py
```

This script:
1. Loads E0 and E1 models from saved checkpoints
2. Computes ROC selectivity for all factors
3. Analyzes weight-selectivity correlations
4. Fits effective connectivity models
5. Runs graded ablation analysis
6. Generates all figures and this report

---

## Configuration Parameters

```python
CONFIG = {
    'data_path': 'data/rnn_export_Newton_08_15_2025_SC.mat',
    'visual_window_start_ms': 50.0,
    'visual_window_end_ms': 250.0,
    'min_trials': 5,
    'n_perm': 200,          # Permutations for p-value
    'n_bootstrap': 200,     # Bootstrap samples for CI
    'input_embed_dim': 56,
    'input_embed_type': 'attention',  # E0
    'attention_heads': 4,
    'learnable_h0': True,
    'tau': 50.0,
    'dt': 25.0,
}
```

---

## Related Documents

- **Biological input experiments overview:** `results/biological_input_experiments/BIOLOGICAL_EXPERIMENTS_REPORT.md`
- **Input restructuring spec:** `specs/INPUT_RESTRUCTURING_SPEC.md`
- **Original neuronal analysis pipeline:** `4factors-analysis-pipeline/code/utils/arrayROC.m`

---

*Report generated: 2026-02-01*
