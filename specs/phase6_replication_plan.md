# Phase 6: Cross-Session Replication Plan

## Overview

This plan documents the replication study to test whether the key findings from Newton_08_15_2025_SC generalize across sessions from the same animal (Newton/Animal 1).

## Original Findings (Newton_08_15_2025_SC)

| Metric | Value |
|--------|-------|
| Validation PSTH correlation | 0.407 |
| Mean Gini coefficient | 0.360 |
| Mean effective targets | 35.7/52 (69%) |
| Gini vs shuffle p-value | 0.712 |
| Entropy vs shuffle p-value | 0.702 |
| h0 E/I difference p-value | 0.0024 (I > E) |
| Factor-specific targeting | Not significant (all p > 0.05) |
| I neuron clustering | Not significant (p = 0.504) |

**Key conclusions:**
1. Inhibition is broadly distributed (global), not factor-specific
2. I neurons have significantly higher h0 than E neurons
3. No evidence for factor-specific I→E targeting at group level

---

## Datasets to Process

| Dataset | File | Size | Status |
|---------|------|------|--------|
| Newton_08_15_2025_SC | rnn_export_Newton_08_15_2025_SC.mat | 7.1 MB | **Original (complete)** |
| Newton_08_14_2025_SC | rnn_export_Newton_08_14_2025_SC.mat | 3.1 MB | To be trained |
| Newton_08_13_2025_SC | rnn_export_Newton_08_13_2025_SC.mat | 1.1 MB | To be trained |

**Note:** File sizes suggest different numbers of trials per session. This is expected and will be documented.

---

## Training Configuration

Use identical configuration to the original model (train_final_model.py):

```python
CONFIG = {
    # Training parameters
    'max_epochs': 2000,
    'patience': 300,
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,
    'seed': 42,

    # Model parameters
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,
    'learnable_h0': True,
    'h0_init': 0.1,

    # Loss parameters
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'lambda_reg': 1e-4,
    'lambda_scale': 0.1,
    'lambda_var': 0.05,
}
```

---

## Analyses to Run

For each dataset, run the full connectivity analysis (same as analyze_connectivity.py):

### 1. Factor Selectivity
- Compute partial eta-squared for each factor (reward, location, identity, salience)
- Compare E vs I neuron selectivity
- Multiple regression with standardized coefficients

### 2. I→E Connectivity Structure
- Extract I→E weight submatrix
- Per-I neuron statistics (mean weight, Gini, effective targets)
- Verify all weights are inhibitory

### 3. I Neuron Clustering
- K-means clustering (k=2,3,4)
- Hierarchical clustering
- Silhouette score analysis
- Permutation test for cluster significance
- PCA visualization

### 4. Factor-Specific Targeting
- Correlate |I→E weights| with E neuron factor selectivity
- Group-level t-tests
- Per-I neuron significance

### 5. Inhibition Specificity
- Gini coefficient (inequality of weight distribution)
- Normalized entropy
- Effective number of targets (inverse participation ratio)
- Shuffle control comparison

### 6. Input Weight Analysis
- E vs I input weight comparison
- FDR-corrected significance tests

### 7. h0 Analysis
- E vs I h0 comparison
- Correlation with factor selectivity

---

## Comparison Metrics

Cross-session comparison will focus on:

### Primary Metrics (must replicate)
| Metric | Expectation | Test |
|--------|-------------|------|
| Validation correlation | > 0.3 | Per-session |
| Gini vs shuffle | p > 0.05 (not different) | Per-session |
| Effective targets | > 50% of E neurons | Per-session |
| h0 E < I | p < 0.05 | Per-session + meta-analysis |

### Secondary Metrics (consistency check)
| Metric | Test |
|--------|------|
| Mean Gini coefficient | Compare across sessions |
| Mean entropy | Compare across sessions |
| Factor-specific correlation | Direction consistency |
| E vs I correlation | Pattern consistency |

### Statistical Approach
1. **Per-session tests:** Run same analyses as original
2. **Cross-session consistency:**
   - Report all 3 sessions' metrics in table
   - Chi-square/Fisher test for categorical outcomes
   - Meta-analysis for h0 E/I difference (combine p-values)
3. **Effect size comparison:** Cohen's d or similar

---

## Expected Outputs

### Per Dataset
```
results/replication/{dataset_name}/
├── model_best.pt                    # Best checkpoint
├── model_final.pt                   # Final checkpoint
├── training_log.json                # Training history
├── config.json                      # Configuration used
├── training_report.md               # Training summary
├── weights/
│   ├── W_rec.npy                    # Recurrent weights
│   ├── W_in.npy                     # Input weights
│   ├── W_out.npy                    # Output weights
│   ├── h0.npy                       # Initial state
│   ├── attention_weights.npy       # Attention parameters
│   ├── E_mask.npy                   # E neuron mask
│   └── I_mask.npy                   # I neuron mask
├── outputs/
│   ├── val_model_rates.npy          # Model firing rates
│   ├── val_target_rates.npy         # Target firing rates
│   ├── val_model_psth.npy           # Trial-averaged model
│   ├── val_target_psth.npy          # Trial-averaged target
│   └── val_trial_conditions.npy     # Condition labels
├── metrics/
│   ├── per_neuron_correlation.npy   # Per-neuron PSTH correlation
│   ├── per_neuron_mean_rate_*.npy   # Mean rates
│   ├── per_neuron_fano_*.npy        # Fano factors
│   └── neuron_ei_labels.npy         # E/I labels
├── population/
│   ├── pca_real.npy                 # PCA of real data
│   ├── pca_model.npy                # PCA of model data
│   └── pca_explained_variance.npy   # Variance explained
├── connectivity_analysis/
│   ├── factor_selectivity_heatmap.png
│   ├── factor_selectivity_ei_comparison.png
│   ├── ie_weight_matrix.png
│   ├── i_neuron_clustering.png
│   ├── cluster_factor_selectivity.png
│   ├── inhibition_specificity.png
│   ├── w_in_heatmap.png
│   ├── h0_vs_selectivity.png
│   ├── summary_figure.png
│   ├── factor_selectivity.csv
│   ├── i_neuron_clusters.csv
│   └── analysis_summary.csv
└── figures/
    ├── training_curves.png
    ├── per_neuron_correlation_histogram.png
    └── ... (other training figures)
```

### Reports
- `specs/Newton_08_14_2025_SC_connectivity_report.md`
- `specs/Newton_08_13_2025_SC_connectivity_report.md`

### Comparison Outputs
```
results/replication/comparison/
├── performance_comparison.png       # Val correlation by session
├── inhibition_specificity_comparison.png  # Gini/entropy comparison
├── h0_comparison.png                # h0 by E/I across sessions
├── ie_weights_comparison.png        # I→E matrices side-by-side
├── factor_selectivity_comparison.png  # Selectivity patterns
├── summary_comparison.png           # Multi-panel summary
├── metrics_table.csv                # All metrics in table
└── statistical_tests.csv            # Cross-session tests
```

### Final Report
- `specs/phase6_replication_report.md` - Complete replication analysis

---

## Timeline and Execution

### Script: run_replication_animal1.py
- **For each dataset:** Train model + run connectivity analysis
- **Estimated time:** ~1-2 hours per dataset (varies with n_trials)
- **Can run unattended:** Yes (progress printed to stdout)

### Script: compare_sessions.py
- **Run after:** Both replications complete
- **Input:** results/final_model/ + results/replication/Newton_08_1{3,4}_2025_SC/
- **Output:** Comparison figures and report

---

## Success Criteria

### Replication Success (all must hold):
1. **Global inhibition replicates:** Gini not significantly different from shuffled in all 3 sessions
2. **h0 E/I difference replicates:** I > E in at least 2/3 sessions, meta-analysis p < 0.05
3. **No factor-specific targeting:** No consistent significant correlations across sessions

### Partial Replication:
- If 2/3 sessions show consistent patterns
- Or if meta-analysis supports main conclusions

### Failed Replication:
- If patterns are inconsistent across sessions
- If original finding was session-specific artifact

---

## Next Steps After Replication

Based on results:

| Outcome | Next Step |
|---------|-----------|
| Full replication | Proceed to Animal 2 (Feynman) |
| Partial replication | Pool sessions for increased power |
| Failed replication | Investigate session differences |

---

## Implementation Notes

1. **Data-specific adjustments:** Each dataset may have different n_neurons. The analysis scripts handle this automatically.

2. **Random seed:** Use same seed (42) for reproducibility.

3. **Progress monitoring:** Scripts print detailed progress updates.

4. **Error handling:** Scripts catch and log errors without stopping.

5. **Memory management:** Clear large arrays after saving to prevent OOM.

---

*Plan created: 2026-01-25*
