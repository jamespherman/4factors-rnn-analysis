# I→E Connectivity Analysis Plan

## Overview

This document outlines the analysis plan for testing whether inhibitory interneurons in the learned SC model show factor-specific structure in their connectivity to excitatory neurons.

## Scientific Background

In the 4-factors task, SC neurons respond to stimuli varying along four dimensions:
1. **Reward magnitude** (binary: high vs low)
2. **Salience** (0=N/A, 1=high, 2=low; bullseye stimuli only)
3. **Probability** (binary: rare vs common)
4. **Identity** (1=face, 2=non-face, 3=bullseye)

**Hypothesis**: Local inhibitory interneurons might be organized to mediate competition along specific factor dimensions. For example, "reward interneurons" might preferentially inhibit excitatory neurons based on reward-related signals, implementing factor-specific gain modulation.

## Data Summary

- **Neurons**: 65 total (52 E, 13 I)
  - Recorded E: 41 (indices 0-40)
  - Hidden E: 11 (indices 41-51)
  - Recorded I: 13 (indices 52-64)
- **Weight matrix**: W_rec (65×65) with Dale's Law constraints
- **Validation trials**: 208 trials with trial conditions
- **Model outputs**: Per-trial firing rates (208×219×54)

---

## Analysis 1: Factor Selectivity of Recorded Neurons

### Scientific Question
How selective are SC neurons (both E and I) for each of the four factors?

### Method
For each of the 54 recorded neurons:
1. **Linear regression approach**: Fit a multiple linear regression model predicting mean firing rate from the 4 factors
   ```
   rate_i = β_0 + β_reward * reward + β_salience * salience + β_probability * probability + β_identity * identity + ε
   ```
2. Compute **partial η²** (effect size) for each factor
3. Alternatively: Compute **Ω² (omega-squared)** from factorial ANOVA

### Statistical Tests
- Multiple regression with F-tests for each predictor
- Bonferroni correction for multiple comparisons (54 neurons × 4 factors)
- Report both p-values and effect sizes (partial η²)

### Output
- **Selectivity matrix**: [54 neurons × 4 factors] containing effect sizes
- **Significance matrix**: [54 neurons × 4 factors] binary significant/not

### Interpretation
- High selectivity index → neuron's firing rate strongly modulated by that factor
- Compare E vs I distributions for each factor

---

## Analysis 2: I→E Connectivity Structure

### Scientific Question
What is the structure of inhibitory projections from I neurons to E neurons?

### Method
1. Extract **I→E submatrix** from W_rec: W_IE = W_rec[0:52, 52:65]
   - Shape: 52 E neurons × 13 I neurons
   - All values should be ≤ 0 (inhibitory)
2. For each I neuron (column), compute:
   - **Mean absolute weight**: How strong is its overall inhibition?
   - **Weight variance**: How variable is it across targets?
   - **Sparsity**: Fraction of very weak connections (|w| < threshold)
   - **Max target count**: Number of E neurons receiving > 50% of max weight

### Output
- I→E weight matrix heatmap
- Per-I neuron statistics table

---

## Analysis 3: Clustering I Neurons by Output Connectivity

### Scientific Question
Are there distinct "types" of I neurons based on their connectivity patterns?

### Methods

#### 3a. Hierarchical Clustering
- Use I→E weight vectors as features (13 vectors of length 52)
- Distance metric: correlation distance (1 - Pearson r)
- Linkage: Ward's method
- Output: dendrogram showing clustering structure

#### 3b. K-means Clustering
- Test k = 2, 3, 4 clusters
- Evaluate with silhouette score and gap statistic
- Select optimal k

#### 3c. PCA Visualization
- Project 13 I neurons into PC space based on their I→E weight profiles
- Visualize clustering structure in 2D

### Statistical Tests
- Compare silhouette scores across k values
- Permutation test: Are clusters more distinct than expected by chance?
  - Shuffle I→E weights 1000 times, re-cluster, compare silhouette scores

### Output
- Dendrogram
- PCA scatter plot with cluster labels
- Optimal cluster assignments

---

## Analysis 4: Relate I Neuron Clusters to Factor Selectivity

### Scientific Question
Do I neuron clusters correspond to factor-specific inhibition?

### Method

#### 4a. E Target Selectivity Analysis
For each I neuron cluster:
1. Identify "strongly inhibited" E neurons (top quartile of |weights|)
2. Identify "weakly inhibited" E neurons (bottom quartile)
3. Compare factor selectivity between strong vs weak targets

#### 4b. I Neuron Self-Selectivity
- Do I neurons within the same cluster share factor selectivity?
- Compute within-cluster vs between-cluster selectivity similarity

#### 4c. Target Selectivity Correlation
For each I neuron:
1. Correlate |I→E weights| with E neuron selectivity for each factor
2. Positive correlation → I neuron preferentially inhibits factor-selective E neurons

### Statistical Tests
- **Wilcoxon rank-sum test**: Compare selectivity distributions (strong vs weak targets)
- **Correlation test**: Spearman correlation between weights and selectivity
- Multiple comparisons correction (FDR)

### Interpretation
- If an I neuron shows strong correlation between its weights and reward-selectivity of targets, it may implement "reward-specific" inhibition

---

## Analysis 5: Local vs Distributed Inhibition

### Scientific Question
Is inhibition targeted (specific E subgroups) or global (broadcast to all)?

### Metrics

#### 5a. Gini Coefficient
- Measures inequality of weight distribution
- Gini = 0: all weights equal (global inhibition)
- Gini = 1: all weight to one target (maximally local)

#### 5b. Entropy
- H = -Σ p_i log(p_i) where p_i = |w_i| / Σ|w|
- High entropy → distributed
- Low entropy → concentrated

#### 5c. Effective Targets
- N_eff = 1 / Σ p_i² (inverse participation ratio)
- Number of E neurons receiving substantial inhibition

### Controls
- **Shuffle control**: Permute I→E weights within each I neuron 1000 times
- **Random weights**: Generate random weight matrices with same marginal statistics
- Compare real Gini/entropy distributions to shuffled

### Statistical Tests
- One-sample t-test: Real Gini vs shuffled mean
- Permutation p-value: Fraction of shuffles with more extreme values

---

## Analysis 6: Input Selectivity (W_in Analysis)

### Scientific Question
What inputs drive E vs I neurons differentially?

### Method
1. Extract W_in (65×56) - input weights
2. Focus on first 14 columns (original inputs before attention embedding):
   - Fixation (1), Target location (4), Go signal (1), Reward (1), Eye position (2), Stimulus features (5)
3. Compare input weight magnitudes between E and I neurons
4. Analyze attention weights separately (if meaningful)

### Statistical Tests
- Mann-Whitney U test for each input: E vs I neurons
- Correction for multiple comparisons

### Interpretation
- If I neurons receive stronger reward input → they may implement reward-modulated inhibition

---

## Analysis 7: Initial State (h0) Patterns

### Scientific Question
Does the learned initial state relate to factor selectivity?

### Method
1. Load h0 (65×1) - learned initial state
2. Correlate h0 with factor selectivity (for recorded neurons)
3. Compare h0 between E and I neurons

### Output
- Scatter plots: h0 vs selectivity for each factor
- Box plot: h0 distribution E vs I

---

## Controls and Statistics Summary

| Analysis | Control | Test | Correction |
|----------|---------|------|------------|
| Factor selectivity | None (descriptive) | F-test, partial η² | Bonferroni |
| I→E structure | Random weights | Descriptive | - |
| Clustering | Shuffled weights | Silhouette, permutation | - |
| Cluster-selectivity | Shuffled cluster labels | Wilcoxon, Spearman | FDR |
| Local vs distributed | Shuffled weights | t-test, permutation | - |
| Input selectivity | None | Mann-Whitney | Bonferroni |
| h0 patterns | None | Spearman correlation | - |

---

## Expected Outputs

### Figures (8-10 figures)
1. `factor_selectivity_heatmap.png` - 54 neurons × 4 factors
2. `factor_selectivity_ei_comparison.png` - Violin plots by E/I
3. `ie_weight_matrix.png` - I→E weights (sorted)
4. `i_neuron_clustering.png` - Dendrogram + PCA
5. `cluster_factor_selectivity.png` - Selectivity by cluster
6. `inhibition_specificity.png` - Gini/entropy vs controls
7. `w_in_heatmap.png` - Input weights for E vs I
8. `attention_weights_visualization.png` - Attention patterns
9. `h0_vs_selectivity.png` - Initial state analysis

### Tables
1. Per-neuron factor selectivity (CSV)
2. I neuron cluster assignments
3. Statistical test results summary

### Report Sections
1. Executive summary (3-5 bullets)
2. Factor selectivity results
3. I→E connectivity structure
4. Factor-specificity of inhibition
5. Local vs distributed inhibition
6. Input organization
7. Conclusions and limitations

---

## Possible Outcomes and Interpretations

### Scenario A: Factor-Specific I Neurons
**Finding**: I neurons cluster into distinct groups, and each cluster preferentially inhibits E neurons selective for different factors.

**Interpretation**: The model learned factor-specific inhibitory circuits. I neurons may implement competitive normalization along specific feature dimensions (e.g., "reward interneurons" mediate reward-based competition).

**Implication**: Supports the hypothesis that SC interneurons have functional specialization for specific stimulus features.

### Scenario B: Distributed Inhibition
**Finding**: No clear I neuron clusters; all I neurons inhibit similar E neuron subsets; no relationship between I→E weights and factor selectivity.

**Interpretation**: Inhibition serves a global normalization function rather than factor-specific modulation. The model uses I neurons for general gain control.

**Implication**: SC interneurons may implement divisive normalization broadly, with factor-specific processing happening elsewhere.

### Scenario C: Mixed Results
**Finding**: Some I neurons show factor-specificity, others don't; partial clustering structure.

**Interpretation**: Heterogeneous inhibitory population with both specialized and generalist interneurons. May reflect a combination of local feature-specific and global normalization mechanisms.

**Implication**: SC may have multiple classes of interneurons with different computational roles.

### Scenario D: Spatial/Location-Specific Inhibition
**Finding**: I neurons cluster by location selectivity of their targets rather than by reward/identity/salience.

**Interpretation**: Inhibition is organized topographically (by visual field location) rather than by non-spatial features.

**Implication**: SC interneurons implement spatial competition (winner-take-all within spatial locations) rather than feature-based competition.

---

## Caveats and Limitations

1. **Model limitations**: The learned connectivity reflects the model's solution, which may not match biological reality. Multiple network configurations could produce similar output.

2. **Limited I neurons**: Only 13 I neurons limits statistical power for clustering and group comparisons.

3. **Factor confounds**: Some factors are not fully crossed (e.g., salience only applies to bullseye stimuli).

4. **Hidden E neurons**: 11 hidden E neurons have no ground-truth selectivity data; we can only analyze connectivity, not function.

5. **Causal interpretation**: Correlation between I→E weights and selectivity doesn't prove causal factor-specific inhibition - it may reflect other organizational principles.

6. **Training objective**: The model was trained to match firing rates, not to implement biologically plausible circuits. The learned structure reflects the optimization landscape.
