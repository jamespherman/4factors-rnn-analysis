# Condition-Specific PSTH Loss Implementation Plan

**Date**: 2026-01-26
**Status**: Implementation Plan

## 1. Problem Statement

### Current Implementation
The current PSTH loss function (`compute_L_neuron()` in `src/losses.py`) computes a **grand-average** across ALL trials:

```python
model_psth = model_rates.mean(dim=0)  # Average ALL trials together
target_psth = target_rates.mean(dim=0)
```

### Fundamental Limitation
This approach washes out **factor selectivity**. Since all trials are averaged together:
- High-reward and low-reward trials are combined
- High-salience and low-salience trials are combined
- Different target locations are combined

The model has **no incentive** to produce different responses for different experimental conditions. It can achieve low PSTH loss by:
1. Producing the same average response for all conditions
2. Finding "degenerate" solutions that match population-level activity without preserving individual neuron tuning

### Observable Consequences
1. **L_trial plateau**: The trial-matching loss never decreases substantially because the model produces identical (or nearly identical) responses for all trials
2. **Poor selectivity matching**: RNN neuron selectivity doesn't correlate with recorded neuron selectivity
3. **Degenerate solutions**: Model weights don't reflect true factor-specific inhibition structure

## 2. Proposed Solution

### Condition-Specific PSTH Loss
Instead of computing one grand-average PSTH, compute **separate PSTHs for each experimental condition** and enforce matching for each.

### Condition Structure
The experimental task has a factorial design:
- **4 target locations** (quadrants)
- **2 reward levels** (high/low)
- **2 salience levels** (high/low)

This gives **4 × 2 × 2 = 16 conditions** total.

### Loss Formulation
For each condition `c`:
1. Select trials belonging to condition `c`
2. Compute condition-specific PSTH for model: `model_psth_c = mean(model_rates[condition == c])`
3. Compute condition-specific PSTH for target: `target_psth_c = mean(target_rates[condition == c])`
4. Compute correlation loss between `model_psth_c` and `target_psth_c`

Final loss = mean across all conditions (equal weighting).

### Mathematical Formulation
Let `C` be the set of all conditions, and `T_c` be the set of trials in condition `c`:

```
L_neuron_conditioned = (1/|C|) * Σ_c [L_corr(mean(model[T_c]), mean(target[T_c]))]
```

where `L_corr` is the correlation-based loss from the existing `compute_L_neuron()` function.

## 3. Implementation Details

### 3.1 New Function in `src/losses.py`

```python
def compute_L_neuron_conditioned(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    trial_conditions: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    lambda_scale: float = 0.1,
    lambda_var: float = 0.05,
    min_trials_per_condition: int = 5
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """
    Compute PSTH loss separately for each experimental condition.

    Args:
        model_rates: [n_trials, n_time, n_neurons] model firing rates
        target_rates: [n_trials, n_time, n_neurons] target firing rates
        trial_conditions: [n_trials] integer condition label for each trial
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [n_trials, n_time] validity mask
        recorded_indices: Which model neurons correspond to recorded neurons
        lambda_scale: Weight for scale matching loss
        lambda_var: Weight for variance matching loss
        min_trials_per_condition: Minimum trials required per condition

    Returns:
        loss: Mean loss across all conditions
        per_condition_loss: Dict of {condition: loss} for logging
    """
```

### 3.2 Data Loader Updates (`src/data_loader.py`)

Add method to `RNNDataset`:

```python
def get_condition_labels(self) -> np.ndarray:
    """
    Create condition labels from factorial design.

    Conditions are indexed as:
        condition = location * 4 + reward * 2 + salience

    where:
        - location: 0-3 (from trial_location 1-4)
        - reward: 0-1 (low/high)
        - salience: 0-1 (low/high)

    Returns:
        [n_trials] array of condition indices (0-15)
    """
```

Also update `get_all_trials()` to include condition labels.

### 3.3 Training Script Changes

Create `scripts/train_conditioned_loss.py`:
- Replace `compute_L_neuron()` with `compute_L_neuron_conditioned()`
- Keep gradient-balanced Sinkhorn L_trial
- Add logging for per-condition losses
- Add selectivity analysis during validation

## 4. Training Experiment Design

### 4.1 Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_epochs | 2000 | Same as final model |
| patience | 300 | Same as final model |
| lr | 1e-3 | Same as final model |
| architecture | attention + h0 | Best from Phase 5 |
| loss weights | gradient-balanced | Proven effective |

### 4.2 Data Split
- Same 80/20 train/val split as final model
- Stratify by condition to ensure balanced representation

### 4.3 Logging
Every epoch:
- Overall validation correlation
- L_trial value
- Learning rate

Every 100 epochs:
- Per-condition validation correlations
- Per-factor selectivity correlation (recorded vs RNN)
- Condition trial counts

### 4.4 Output Directory
```
results/conditioned_loss_08_15/
├── model_best.pt
├── model_final.pt
├── training_log.json
├── config.json
├── weights/
├── outputs/
├── metrics/
├── population/
├── figures/
└── checkpoints/
```

## 5. Evaluation Metrics

### 5.1 Primary Metric: Selectivity Matching
For each factor (location, reward, salience):
1. Compute selectivity index (ROC or partial eta-squared) for each neuron in:
   - Recorded data
   - RNN model (conditioned loss)
   - RNN model (original grand-average loss, from results/final_model/)
2. Compute Pearson correlation between recorded and RNN selectivity
3. Compare conditioned vs original model

**Success criterion**: Selectivity correlation improves for at least 2 of 3 factors.

### 5.2 Secondary Metric: L_trial Trajectory
- Plot L_trial over training epochs
- Compare to original model's L_trial trajectory
- **Expected**: L_trial should now decrease during training since condition-specific responses create natural trial-to-trial variability

### 5.3 Tertiary Metric: Overall PSTH Correlation
- Compare validation PSTH correlation to original model
- **Expected**: May decrease slightly since we're now fitting 16 condition-specific patterns instead of 1 grand-average
- **Acceptable**: Within 0.05 of original (e.g., 0.72 → 0.67)

### 5.4 Factor Selectivity Metrics
For each neuron and factor:
1. **ROC-AUC**: Area under ROC curve for discriminating high vs low conditions
2. **Partial eta-squared**: Variance explained by factor in ANOVA
3. **d-prime**: Signal-to-noise ratio for factor discrimination

## 6. Expected Outcomes

### 6.1 Positive Outcomes (If Solution Works)
1. **Selectivity matching improves**: r(recorded, RNN) increases from ~0 to >0.3
2. **L_trial decreases**: Model produces condition-specific variability
3. **Connectivity analysis changes**: I→E weights may show factor-specific targeting
4. **Biological interpretability**: Model weights reflect true neural tuning

### 6.2 Neutral/Negative Outcomes (If Solution Doesn't Work)
1. **No improvement**: May indicate selectivity is not learnable from current inputs
2. **Training instability**: May need different loss weighting
3. **Overfitting to conditions**: May need regularization

### 6.3 Fallback Strategies
If condition-specific loss doesn't help:
1. Try per-trial loss (more fine-grained)
2. Add explicit selectivity regularization
3. Augment input representation with factor information

## 7. Implementation Timeline

| Step | Description | Files |
|------|-------------|-------|
| 1 | Implement `compute_L_neuron_conditioned()` | `src/losses.py` |
| 2 | Add `get_condition_labels()` to data loader | `src/data_loader.py` |
| 3 | Create training script | `scripts/train_conditioned_loss.py` |
| 4 | Run training | (execution) |
| 5 | Evaluate selectivity matching | (analysis) |
| 6 | Run connectivity analysis | `scripts/analyze_connectivity.py` |
| 7 | Write report | `specs/condition_specific_loss_report.md` |
| 8 | Generate figures | (visualization) |

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Unbalanced trial counts per condition | Low | Medium | Check data, weight by trial count |
| Training instability | Medium | Medium | Use gradient balancing, careful hyperparameters |
| Overfitting to small conditions | Low | Medium | Min trials threshold, regularization |
| No improvement | Medium | High | Fallback strategies documented |

## 9. Success Criteria

**Minimum success**:
- Code runs without errors
- Model trains to convergence
- Clear comparison data generated

**Partial success**:
- L_trial decreases during training
- At least 1 factor shows improved selectivity matching

**Full success**:
- Selectivity matching improves for ≥2 factors
- L_trial decreases substantially
- Overall PSTH correlation within 0.05 of original
- Clear improvement in connectivity analysis interpretability
