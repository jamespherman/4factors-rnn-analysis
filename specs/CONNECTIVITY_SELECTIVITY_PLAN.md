# Connectivity-Selectivity Analysis Plan

**Date:** 2026-02-01
**Status:** Implementation

---

## Part 1: Literature Summary

Based on comprehensive literature review, the main approaches for relating connectivity to selectivity include:

### Established Methods

1. **Ablation-based contribution analysis** (Meyes et al. 2019)
   - Remove connections/neurons and measure selectivity change
   - Already implemented in our ablation suite (I→E, E→E, E→I ablations)

2. **Weight clustering analysis** (Song et al. 2016)
   - Sort neurons by selectivity, examine if like-to-like connectivity emerges
   - Expectation: neurons with similar selectivity should have stronger E→E connections

3. **Input weight analysis**
   - Relate W_in weights to selectivity
   - Direct input contribution vs recurrence-derived selectivity

4. **Effective connectivity regression**
   - Regress selectivity on incoming connection weights
   - E.g., neuron i's reward selectivity ~ sum of W_rec[i, j] * neuron j's reward selectivity

### Novel Approaches for This Analysis

5. **Connection-specific selectivity contribution**
   - For each neuron, compute: what fraction of selectivity comes from E→E vs I→E inputs?
   - Method: partial ablations and selectivity change measurement

6. **Selectivity propagation analysis**
   - Track how selectivity "flows" through the network
   - Use input assignment to identify direct vs emergent selectivity

---

## Part 2: ROC-Based Selectivity Implementation

### Design Matching User's Neuronal Pipeline

```python
def compute_roc_auc(rates_cond1: np.ndarray, rates_cond2: np.ndarray) -> float:
    """
    Compute ROC-AUC using Mann-Whitney U statistic.

    AUC = P(X_cond2 > X_cond1) = U / (n1 * n2)

    Returns:
        AUC in [0, 1], where 0.5 = chance
    """
    # Rank-based computation (matches arrayROC.m)
    pooled = np.concatenate([rates_cond1, rates_cond2])
    n1, n2 = len(rates_cond1), len(rates_cond2)

    ranks = scipy.stats.rankdata(pooled)
    rank_sum_2 = ranks[n1:].sum()

    U2 = rank_sum_2 - n2 * (n2 + 1) / 2
    auc = U2 / (n1 * n2)

    return auc
```

### Time Window

Use **target-onset aligned window** matching neuronal analysis:
- SC visual epoch: 50-250 ms post-target onset
- With 25ms bins: bins 2-10 after target onset (0-indexed)

### Permutation Testing

```python
def permutation_test(rates_cond1, rates_cond2, observed_auc, n_perm=200):
    """
    Permutation test: shuffle condition labels, recompute AUC.

    Returns:
        p_value: fraction of permuted AUCs as extreme as observed
    """
    pooled = np.concatenate([rates_cond1, rates_cond2])
    n1 = len(rates_cond1)

    null_aucs = []
    for _ in range(n_perm):
        perm_idx = np.random.permutation(len(pooled))
        perm_auc = compute_roc_auc(pooled[perm_idx[:n1]], pooled[perm_idx[n1:]])
        null_aucs.append(perm_auc)

    # Two-sided p-value
    observed_deviation = abs(observed_auc - 0.5)
    p_value = np.mean(np.abs(np.array(null_aucs) - 0.5) >= observed_deviation)

    return p_value
```

---

## Part 3: Connectivity-Selectivity Analysis Design

### Analysis 1: Weight-Selectivity Correlation

**Question:** Do neurons with similar selectivity have stronger connections?

**Method:**
1. Compute ROC-AUC selectivity for each neuron (for each factor)
2. For each pair of neurons (i, j), compute:
   - Selectivity similarity: |AUC_i - AUC_j| or correlation
   - Connection strength: |W_rec[i, j]|
3. Correlate selectivity similarity with connection strength
4. Separate by connection type (E→E, E→I, I→E, I→I)

**Expected Result:**
- E→E connections: positive correlation (like-to-like)
- I→E connections: may show anti-correlation (competitive inhibition)

### Analysis 2: Input Weight Contribution

**Question:** How much does direct input vs recurrence contribute to selectivity?

**Method:**
1. For each neuron, identify which factors it receives (from InputAssignment)
2. Compute correlation between W_in weights and factor selectivity
3. Compare neurons that DO vs DON'T receive direct input

### Analysis 3: Effective Connectivity Regression

**Question:** Can we predict a neuron's selectivity from its inputs?

**Method:**
For neuron i and factor f:
```
selectivity_i = β0 + β_E * Σ(W_rec[i,j] * selectivity_j for j in E)
                   + β_I * Σ(W_rec[i,j] * selectivity_j for j in I)
                   + β_direct * receives_input[i,f]
```

This decomposes selectivity into:
- β_E: E→E contribution
- β_I: I→E contribution
- β_direct: Direct input contribution

### Analysis 4: Graded Ablation Analysis

**Question:** How does selectivity change as we gradually reduce connection strength?

**Method:**
1. Scale connection blocks by factors [0, 0.25, 0.5, 0.75, 1.0]
2. Recompute selectivity at each level
3. Plot selectivity vs connection strength
4. Separate by factor type (spatial vs feature)

### Analysis 5: Selectivity Flow Visualization

**Question:** How does selectivity "propagate" through the network?

**Method:**
1. Identify "source" neurons (receive direct factor input)
2. Identify "target" neurons (emergent selectivity, no direct input)
3. Visualize the connection pathways from sources to targets
4. Weight edges by connection strength
5. Color nodes by selectivity magnitude

---

## Part 4: Implementation Plan

### Step 1: Create ROC Selectivity Module

File: `src/analysis/roc_selectivity.py`
- `compute_roc_auc()`: Mann-Whitney U based AUC
- `compute_roc_selectivity()`: ROC for all neurons, single factor
- `compute_all_roc_selectivities()`: All factors
- `permutation_test_roc()`: Permutation-based significance
- `find_visual_epoch_bins()`: Target-onset aligned window

### Step 2: Create Connectivity Analysis Module

File: `src/analysis/connectivity_selectivity.py`
- `analyze_weight_selectivity_correlation()`: Analysis 1
- `analyze_input_weight_contribution()`: Analysis 2
- `fit_effective_connectivity_model()`: Analysis 3
- `graded_ablation_analysis()`: Analysis 4
- `compute_selectivity_flow()`: Analysis 5

### Step 3: Create Visualization Module

File: `src/analysis/connectivity_visualization.py`
- `plot_weight_selectivity_scatter()`: Weight vs selectivity scatter
- `plot_connection_selectivity_heatmap()`: Heatmap of W_rec colored by selectivity
- `plot_selectivity_flow_graph()`: Network visualization
- `plot_graded_ablation_curves()`: Selectivity vs ablation level
- `plot_effective_connectivity_coefficients()`: β coefficients with CIs

### Step 4: Run Analysis Script

File: `scripts/analyze_connectivity_selectivity.py`
- Load trained models (E0, E1)
- Compute ROC selectivity for all neurons
- Run all connectivity analyses
- Generate figures
- Create report

---

## Part 5: Expected Outputs

### Figures
1. `roc_selectivity_comparison.png` - ROC vs correlation selectivity
2. `weight_selectivity_scatter_by_type.png` - 4-panel scatter (E→E, E→I, I→E, I→I)
3. `connection_selectivity_heatmap.png` - W_rec with selectivity overlay
4. `graded_ablation_curves.png` - Selectivity vs ablation level
5. `effective_connectivity_coefficients.png` - β_E, β_I, β_direct
6. `selectivity_flow_network.png` - Network graph visualization

### Report
`results/biological_input_experiments/CONNECTIVITY_ANALYSIS_REPORT.md`
- ROC selectivity results
- Connectivity-selectivity correlations
- Effective connectivity model fits
- Interpretation in terms of hypothesis

---

## Part 6: Success Criteria

1. **ROC selectivity computed** for all neurons matching user's neuronal pipeline
2. **Significant correlation** between E→E weights and selectivity similarity
3. **Differential I→E vs E→E effects** on spatial vs feature factors
4. **Effective connectivity model** explains >30% of selectivity variance
5. **Clear visualizations** relating connection structure to selectivity patterns

---

*Plan created: 2026-02-01*
