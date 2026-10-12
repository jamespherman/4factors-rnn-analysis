# Biologically-Constrained Input Experiments Plan

**Date:** 2026-01-31
**Status:** Planning Phase

---

## Part 1: Audit Synthesis

### Key Findings from 98 Experiments

The comprehensive audit revealed several critical lessons for this work:

#### What Definitively Works
| Technique | Effect | Confidence |
|-----------|--------|------------|
| Attention-based input embedding (56-dim, 4 heads) | +32.5% | HIGH |
| Learnable h0 (init 0.1) | +4.6% | HIGH |
| Gradient balancing (scale 0.5) | +9.5% | HIGH |
| Tight Sinkhorn (epsilon 0.1) | +5% rel | HIGH |
| Extended training (1000-2000 epochs) | +2.5% | HIGH |
| ReduceLROnPlateau scheduler | best | MEDIUM |

#### What Definitively Hurts
| Technique | Effect | Confidence |
|-----------|--------|------------|
| Fixed lambda for L_trial | -15% to -48% | HIGH |
| Soft Sinkhorn (epsilon >= 0.5) | -10% | HIGH |
| Combining multiple improvements | -5% to -20% | HIGH |
| Large embeddings (>56 dim) | -8% to -17% | HIGH |
| Learnable alpha with attention | -3% | MEDIUM |

#### Critical Pattern: Combinations Hurt
Every major combination tested underperformed individual techniques:
- alpha + poisson: 0.3083 vs 0.3449 (alpha alone)
- alpha + low_rank: 0.2782 vs 0.3449
- h0 + AdamW: 0.3987 vs 0.4021 (h0 alone)

**Implication for this work:** The biological input restructuring is ONE modification. We must not add other changes simultaneously.

### Previous Input Restructuring Results

| Experiment | Val Corr | vs Baseline |
|------------|----------|-------------|
| E0 (baseline) | 0.1646 | 100% |
| E1 (temporal fix) | 0.1763 | 107% |
| E2 (primary) | 0.1585 | 96% |

**Key issue:** These experiments used only 42 neurons and did NOT use attention embedding or learnable h0. The fit quality (0.16) was much lower than our best (0.41).

### Emergent Selectivity from Previous Work

Despite the limited configuration, E2 showed emergent selectivity:
- Reward: 50% of non-receivers showed significant selectivity
- Probability: 84% of non-receivers showed significant selectivity
- Salience: 34% of non-receivers showed significant selectivity
- Identity: 41% of non-receivers showed significant selectivity

### Ablation Results from Previous Work

| Ablation | Reward | Probability | Salience | Identity |
|----------|--------|-------------|----------|----------|
| Full recurrence (W_rec=0) | -29% | -50% | +167% | +142% |
| I->E only | +83% | +53% | -83% | -51% |

**Interpretation:**
- Removing I->E connections *increased* spatial factor (reward/probability) selectivity
- Removing I->E connections *decreased* feature factor (salience/identity) selectivity
- This is consistent with: E->E propagates spatial factors; I->E suppresses them but enables feature factors

---

## Part 2: Conflicts Between Audit Recommendations and Biological Constraints

### Conflict 1: Input Embedding vs Per-Neuron Inputs

**Problem:** The attention embedding expects global inputs (14 channels shared across all neurons). Biological constraints require neuron-specific input assignments.

**Resolution Options:**
1. **Pre-embedding masking:** Apply biological masks before attention embedding
2. **Post-embedding masking:** Apply masks after embedding (loses biological specificity)
3. **Per-neuron input channels:** Expand inputs to neuron-specific before embedding
4. **Disable attention embedding:** Use linear embedding or no embedding

**Recommended resolution:** Option 1 (pre-embedding masking). The input tensor will be [n_trials, n_bins, n_neurons, n_input_channels], and we mask before flattening for the embedding. This preserves both the attention benefit and biological constraints.

However, this requires architectural changes. A simpler approach:
- Use **linear embedding instead of attention** for this experiment
- This is still a proven technique (+29.7% over baseline)
- Avoids adding complexity while testing biological hypothesis

### Conflict 2: Input Channel Count Changes

**Problem:** Standard input has 14 channels. Restructured input has 6 channels per neuron (reward, probability, target, salience, identity, fixation).

**Resolution:** Adapt input embedding dimension:
- Previous: 14 channels -> 56 dim embedding (4x)
- New: 6 channels -> 24 dim embedding (4x) OR 6 channels -> 28 dim (flexible)

### Conflict 3: I Neurons Receiving Different Inputs

**Problem:** In restructured design, I neurons receive different inputs than E neurons (no RF-matched spatial). This creates heterogeneity that attention may not handle well.

**Resolution:** Use separate input matrices for E and I, or flatten into a larger shared input space.

### Final Resolution Strategy

To minimize risk while testing the biological hypothesis:

1. **Use the 08_15 dataset** (54 neurons: 41E, 13I) - NOT the 42-neuron filtered set
2. **Use linear embedding** (not attention) - still proven effective
3. **Keep learnable h0** - independent improvement
4. **Keep gradient balancing** - essential for L_trial
5. **Implement biological input fractions as per-neuron masks**

---

## Part 3: Biological Input Constraints

### Anatomical Basis

| Pathway | E neurons receiving | I neurons receiving | Signal type |
|---------|---------------------|---------------------|-------------|
| SNr (substantia nigra pars reticulata) | ~28% | ~39% | Value/reward |
| V1 (primary visual cortex) | 67-85% | 11-33% | Visual features |

### Proposed Input Fractions

| Factor | E fraction | I fraction | Rationale |
|--------|------------|------------|-----------|
| Reward | 0.30 | 0.40 | SNr pathway |
| Probability | 0.30 | 0.40 | SNr pathway |
| Salience | 0.75 | 0.25 | V1 pathway |
| Identity | 0.75 | 0.25 | V1 pathway |

**Key asymmetry:**
- Feature factors (salience, identity) preferentially reach E neurons (cortical)
- Value factors (reward, probability) reach both, slight I preference (basal ganglia)

---

## Part 4: Hypothesis and Predictions

### The Hypothesis

**Local SC computation - specifically E->E recurrence - is sufficient for encoding priority signals. The architecture of local I->E inhibition is inconsistent with factor-specific competition: recorded interneurons display response properties consistent with mediating spatial (not factor) competition.**

### Specific Predictions

1. **E->E propagates spatial/value factors (reward, probability)**
   - Test: E->E ablation should reduce emergent reward/probability selectivity
   - Previous result: Full recurrence ablation reduced reward selectivity by 29%

2. **I->E suppresses (rather than creates) spatial factor selectivity**
   - Test: I->E ablation should INCREASE reward/probability selectivity
   - Previous result: I->E ablation increased reward selectivity by 83%

3. **I->E enables feature factor selectivity (salience, identity)**
   - Test: I->E ablation should DECREASE salience/identity selectivity
   - Previous result: I->E ablation decreased salience selectivity by 83%

4. **Emergent selectivity arises in neurons not receiving direct input**
   - Test: With biological fractions, neurons should show selectivity for factors they don't receive
   - The fraction showing emergent selectivity should depend on which connections remain

5. **E neurons should show more emergent spatial selectivity than I neurons**
   - Reason: E neurons receive RF-matched spatial input and can propagate via E->E

6. **I neurons should show more emergent feature selectivity**
   - Reason: I neurons receive more feature input and can propagate via I->E->I loops

---

## Part 5: Experimental Design

### Dataset

**Use:** `data/rnn_export_Newton_08_15_2025_SC.mat`
- 54 neurons (41 E, 13 I)
- 1043 trials (835 train, 208 val with 80/20 split)
- Best previous fit: 0.41 correlation

### Model Configuration

Using the proven best configuration from audit:

```python
CONFIG = {
    # Training
    'max_epochs': 2000,
    'patience': 300,
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,
    'seed': 42,

    # Model
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,

    # MODIFIED: Use linear embedding instead of attention
    # for compatibility with per-neuron input structure
    'input_embed_dim': 28,  # 7 input channels x 4
    'input_embed_type': 'linear',  # Changed from 'attention'

    # Keep these
    'learnable_h0': True,
    'h0_init': 0.1,

    # Loss
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'sinkhorn_epsilon': 0.1,
    'lambda_reg': 1e-4,
}
```

### Experiments

#### Experiment 0: Baseline (Full Input)
- All neurons receive all factors
- Standard 14-channel input
- Attention embedding (best config)
- Purpose: Establish performance ceiling

#### Experiment 1: Biological Input Fractions (Primary)
- E neurons: 30% receive reward/prob, 75% receive salience/identity
- I neurons: 40% receive reward/prob, 25% receive salience/identity
- All neurons receive spatial (target location) and fixation
- Linear embedding (for compatibility)

#### Experiment 2: Biological + Attention (if E1 works)
- Same fractions as E1
- Test if attention can work with restructured inputs
- May require architectural changes

#### Experiment 3: Extreme Constraints (Sensitivity Test)
- E neurons: 100% receive features, 0% receive values
- I neurons: 100% receive values, 0% receive features
- Test whether model can still fit

### Ablation Analyses (for each experiment)

1. **Full recurrence ablation** (W_rec = 0)
   - Tests: Does emergent selectivity require recurrence?

2. **I->E ablation** (W_rec[:n_exc, n_exc:] = 0)
   - Tests: Does I->E suppress spatial or enable features?

3. **E->E ablation** (W_rec[:n_exc, :n_exc] = 0)
   - Tests: Does E->E propagate spatial/value factors?

4. **E->I ablation** (W_rec[n_exc:, :n_exc] = 0)
   - Tests: Are E->I->E loops important?

### Metrics

| Metric | What it tests |
|--------|---------------|
| Val PSTH correlation | Overall fit quality |
| Per-factor emergent selectivity | Recurrence-dependent computation |
| E vs I emergent selectivity | Cell-type specific computation |
| Ablation delta selectivity | Which connections matter for which factors |
| Spatial vs feature factor patterns | Hypothesis about pathway specialization |

---

## Part 6: Implementation Plan

### Phase 1: Input Assignment Module Extensions

Modify `src/input_assignment.py` to support:
1. Biologically-motivated fractions (not just binary assignment)
2. Per-neuron, per-factor fraction specification
3. E vs I specific fractions

New function:
```python
def create_biological_assignment(
    n_neurons: int,
    neuron_type: np.ndarray,
    rf_locations: np.ndarray,
    e_fractions: dict,  # {'reward': 0.3, 'probability': 0.3, 'salience': 0.75, 'identity': 0.75}
    i_fractions: dict,  # {'reward': 0.4, 'probability': 0.4, 'salience': 0.25, 'identity': 0.25}
    seed: int = 0
) -> InputAssignment
```

### Phase 2: Data Loader Extensions

Modify `src/data_loader.py`:
1. Support new input tensor shape [n_trials, n_bins, n_neurons, n_input_channels]
2. Add biological input construction function
3. Ensure compatibility with model forward pass

### Phase 3: Model Modifications

Modify `src/model.py`:
1. Support per-neuron input if needed
2. Add E->E ablation to ablation module
3. Ensure W_in can handle variable input structure

### Phase 4: Training Script

Create `scripts/run_biological_input_experiments.py`:
1. Run baseline experiment
2. Run biological input experiments
3. Run all ablations
4. Compute and save emergent selectivity
5. Generate figures

### Phase 5: Analysis and Reporting

Create `scripts/analyze_biological_experiments.py`:
1. Compare fit quality across conditions
2. Analyze emergent selectivity by factor and neuron type
3. Analyze ablation effects
4. Generate hypothesis-testing figures

---

## Part 7: Expected Results and Interpretation

### Fit Quality Predictions

| Experiment | Expected Val Corr | Rationale |
|------------|-------------------|-----------|
| Baseline (full input) | ~0.41 | Matches best previous |
| Biological fractions | 0.35-0.40 | Some fit loss from reduced input |
| Extreme constraints | 0.30-0.35 | More fit loss |

**Success criterion:** Biological constraints achieve >= 85% of baseline fit (>= 0.35).

### Emergent Selectivity Predictions

| Factor | Who should show emergent? | Via which connections? |
|--------|---------------------------|------------------------|
| Reward | E neurons not receiving reward | E->E |
| Probability | E neurons not receiving prob | E->E |
| Salience | I and E neurons | E->E and I->E loops |
| Identity | I and E neurons | E->E and I->E loops |

### Ablation Predictions

| Ablation | Effect on Reward | Effect on Salience |
|----------|------------------|-------------------|
| Full recurrence | Large decrease | Increase or small decrease |
| I->E only | Increase | Large decrease |
| E->E only | Large decrease | Small change |
| E->I only | Small change | Moderate decrease |

### Hypothesis Evaluation Criteria

The hypothesis is **SUPPORTED** if:
1. E->E ablation reduces reward/probability emergent selectivity more than I->E ablation
2. I->E ablation reduces salience/identity emergent selectivity more than E->E ablation
3. I->E ablation INCREASES reward/probability selectivity (as in previous results)
4. Emergent selectivity patterns differ between E and I neurons consistent with their input profiles

The hypothesis is **REFUTED** if:
1. I->E ablation reduces all factor selectivity equally
2. E->E ablation has no differential effect on spatial vs feature factors
3. I neurons show factor-specific emergent selectivity inconsistent with receiving factor information

---

## Part 8: Output Structure

```
results/biological_input_experiments/
├── analysis_synthesis.md          # This analysis
├── experiments/
│   ├── E0_baseline/
│   │   ├── config.json
│   │   ├── model_best.pt
│   │   ├── training_log.json
│   │   ├── metrics/
│   │   ├── weights/
│   │   └── figures/
│   ├── E1_biological_v1/
│   │   └── [same structure]
│   ├── E2_biological_attention/
│   │   └── [same structure]
│   └── ablations/
│       ├── E0_full_recurrence/
│       ├── E0_ie_only/
│       ├── E0_ee_only/
│       ├── E1_full_recurrence/
│       ├── E1_ie_only/
│       └── E1_ee_only/
├── figures/
│   ├── fit_quality_comparison.png
│   ├── emergent_selectivity_by_factor.png
│   ├── emergent_selectivity_e_vs_i.png
│   ├── ablation_by_factor.png
│   ├── ie_vs_ee_contribution.png
│   └── hypothesis_summary.png
└── REPORT.md
```

---

## Part 9: Success Criteria Summary

1. **Fit quality:** Biological constraints achieve >= 0.35 correlation (>= 85% of baseline)

2. **Emergent selectivity detected:**
   - >= 30% of non-receivers show significant selectivity for each factor
   - Separately demonstrated for E and I neurons

3. **Recurrence dependence:**
   - Full ablation reduces emergent selectivity by >= 30%
   - Different ablation types show differential effects

4. **Hypothesis-specific:**
   - E->E ablation has larger effect on reward than I->E ablation
   - I->E ablation has larger effect on salience than E->E ablation
   - I->E ablation increases (not decreases) reward selectivity

5. **Interpretability:**
   - Results can be explained in terms of the scientific hypothesis
   - Clear dissociation between connection types and factor types

---

## Part 10: Implementation Checklist

- [ ] Create biological input assignment function
- [ ] Extend data loader for per-neuron inputs
- [ ] Verify model compatibility with new input structure
- [ ] Run E0 baseline experiment
- [ ] Run E1 biological fractions experiment
- [ ] Run E2 (if E1 successful)
- [ ] Run all ablations for E0 and E1
- [ ] Compute emergent selectivity for all conditions
- [ ] Generate comparison figures
- [ ] Create final report with hypothesis evaluation

---

*Plan created: 2026-01-31*
