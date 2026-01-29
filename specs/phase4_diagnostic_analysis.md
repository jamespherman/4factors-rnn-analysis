# Phase 4 Diagnostic Analysis

Deep analysis of the best Phase 4 model: **attention_learnable_h0** (val_corr = 0.4021)

## Executive Summary

The learnable initial state (h0) model achieved the Phase 4 target of 0.40 validation PSTH correlation. Diagnostic analysis reveals:

1. **Per-neuron performance is highly variable** - some neurons fit extremely well (r > 0.8), others poorly (r < 0)
2. **I neurons are harder to fit** than E neurons (mean r: 0.15 vs 0.37)
3. **Error accumulates over time** - early timepoints have lower MSE than late
4. **Population structure is well-captured** on PC1 (r = 0.75) but less so on higher PCs
5. **Trial-to-trial variability is severely underestimated** - model Fano factor is ~17x lower than real data
6. **Learned h0 is mostly positive** (92%) with I neurons having higher baseline than E

---

## 1. Per-Neuron Analysis

### Summary Statistics

| Metric | Value |
|--------|-------|
| Mean correlation | 0.317 |
| E neuron mean | 0.370 |
| I neuron mean | 0.150 |
| E neuron std | 0.225 |
| I neuron std | 0.302 |

### Key Findings

**E neurons are significantly easier to fit than I neurons:**
- E neurons (n=41): mean r = 0.370, range [-0.05, 0.83]
- I neurons (n=13): mean r = 0.150, range [-0.31, 0.64]

**High variance across neurons:**
- Best neuron: r = 0.833 (E neuron, index 38)
- Worst neuron: r = -0.306 (I neuron, index 47)
- ~20% of neurons have r < 0.1

**Interpretation:** The model captures some neurons extremely well but struggles with others, particularly inhibitory interneurons. This may reflect:
- Different input selectivity patterns for I neurons
- Insufficient constraint on I neuron dynamics
- Possible need for I-specific loss weighting

### Figures Generated
- `per_neuron_correlation_histogram.png` - Distribution of per-neuron correlations by E/I type
- `e_vs_i_correlation.png` - Boxplot comparison of E vs I performance
- `best_worst_neurons.png` - PSTH traces for best and worst-fit neurons

---

## 2. Temporal Error Analysis

### Summary Statistics

| Metric | Value |
|--------|-------|
| Early MSE (0-20% of trial) | 127.5 |
| Middle MSE (20-80%) | 138.2 |
| Late MSE (80-100%) | 140.8 |
| Mean MSE | 136.6 |
| Mean spatial correlation | 0.199 |

### Key Findings

**Error accumulates over time:**
- Early period has 9% lower MSE than late period
- MSE increases from ~100 at t=30 to ~150 at trial end
- Suggests model dynamics drift from target over long timescales

**Temporal correlation pattern:**
- Spatial correlation (across neurons at each timepoint) averages ~0.20
- Peak correlation around t=30-45 (~0.30), drops to ~0.15 late in trial
- Initial transient period (t<10) has unstable correlation

**Residual analysis:**
- Mean residual is mostly negative in late trial (model underpredicts)
- Strongest negative bias around t=180-200 (mean residual ~ -2.0)
- Small positive bias in early/middle periods

**Interpretation:** The learnable h0 helps with early dynamics but doesn't fully address late-trial drift. The model tends to underestimate firing rates later in trials.

### Figure Generated
- `temporal_error_analysis.png` - Time-resolved MSE, spatial correlation, and residual timecourse

---

## 3. Population Structure (PCA)

### Summary Statistics

| PC | Explained Variance | Model-Real Correlation |
|----|-------------------|----------------------|
| PC1 | 48.6% | 0.750 |
| PC2 | 13.8% | 0.335 |
| PC3 | 4.6% | 0.380 |
| PC4 | 3.3% | 0.642 |
| PC5 | 2.3% | 0.255 |

Top 3 PCs explain 67.0% of variance.

### Key Findings

**Strong capture of dominant mode:**
- PC1 correlation of 0.75 indicates the model captures the main population activity axis
- This dominant mode represents nearly half of population variance

**Weaker capture of secondary structure:**
- PC2-PC3 correlations (~0.34-0.38) suggest partial capture of secondary dynamics
- PC4 surprisingly has good correlation (0.64)
- PC5+ have weak or negative correlations

**State space trajectory:**
- Model trajectory follows real data in PC1-PC2 space
- Some divergence in loop structure and trajectory timing
- Trial onset positions align reasonably well

**Interpretation:** The model captures the main axis of neural activity well but loses fidelity on secondary dimensions. This pattern is typical of models that optimize mean PSTH rather than full neural dynamics.

### Figures Generated
- `pca_trajectories.png` - PC1-3 trajectories over time
- `pca_state_space.png` - 2D state space plots (PC1-PC2, PC2-PC3)

---

## 4. Trial-to-Trial Variability (Fano Factors)

### Summary Statistics

| Metric | Value |
|--------|-------|
| Real mean Fano factor | 48.6 |
| Model mean Fano factor | 2.9 |
| Fano correlation | 0.577 |

### Key Findings

**Model severely underestimates variability:**
- Real Fano factor is ~17x higher than model
- Model generates near-deterministic responses
- Noise scale (0.1) is insufficient to match neural variability

**Relative patterns preserved:**
- Fano correlation of 0.58 suggests model captures which neurons are more/less variable
- But absolute magnitudes are vastly underestimated

**Implications for loss function:**
- L_trial loss measures shape, not amplitude of variability
- Current training does not encourage realistic trial-to-trial variability
- May need explicit Fano factor matching loss or higher noise

### Figures Generated
- `fano_factor_analysis.png` - Scatter plot and histogram of Fano factors
- `fano_factor_time.png` - Time-resolved Fano factors

---

## 5. Initial State (h0) Analysis

### Summary Statistics

| Metric | Value |
|--------|-------|
| h0 E mean | 0.258 |
| h0 E std | 0.204 |
| h0 I mean | 0.387 |
| h0 I std | 0.116 |
| h0 min | -0.407 |
| h0 max | 0.585 |
| Positive fraction | 92.3% |

### Key Findings

**Learned positive baseline:**
- 92% of neurons have positive initial state
- This enables non-zero baseline firing rates (through softplus)
- Model discovered that neural activity starts above zero

**I neurons have higher h0:**
- Mean h0 for I: 0.387 vs E: 0.258
- Lower variance for I (0.116 vs 0.204)
- Suggests I neurons need consistently higher baseline activation

**Range matches expectations:**
- h0 values span [-0.4, 0.6], centered around 0.3
- After softplus, this gives baseline rates of ~0.5-1.0 (before scaling)
- Consistent with non-zero spontaneous activity in real SC neurons

**Interpretation:** The learnable h0 captured meaningful structure about baseline neural activity. The higher I baseline may help drive E neuron activity through the recurrent circuit.

### Figure Generated
- `initial_state_h0.png` - h0 distribution by E/I type and by neuron index

---

## Recommendations for Future Work

Based on this diagnostic analysis:

### High Priority

1. **Address I neuron fitting:**
   - Consider higher weight for I neurons in PSTH loss
   - May need I-specific attention patterns in input embedding
   - Investigate whether I neurons have different input selectivity

2. **Address late-trial error accumulation:**
   - Consider longer training with curriculum on trial length
   - Add explicit temporal consistency regularization
   - May indicate need for better recurrent dynamics

### Medium Priority

3. **Improve variability matching:**
   - Increase noise scale from 0.1 to match real Fano factors
   - Consider explicit Fano factor loss term
   - Current L_trial loss doesn't penalize low variability

4. **Capture secondary population modes:**
   - PC2+ correlations could improve with richer loss functions
   - Consider PCA-space loss in addition to neuron-space

### Lower Priority

5. **Refine initial state:**
   - Current h0 initialization (0.1) works well
   - Could explore neuron-type-specific initialization
   - h0 variance by neuron could be informative

---

## Files Generated

All diagnostic outputs saved to `results/phase4/diagnostics/`:

```
diagnostic_summary.json          # Full numerical results
per_neuron_correlation_histogram.png
e_vs_i_correlation.png
best_worst_neurons.png
temporal_error_analysis.png
pca_trajectories.png
pca_state_space.png
fano_factor_analysis.png
fano_factor_time.png
initial_state_h0.png
```
