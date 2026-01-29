# R01 Figure Plan: RNN Analysis Rejects Local SC Computation

## Overview

This document outlines the plan for generating a publication-quality figure demonstrating that local SC dynamics are insufficient for competition resolution. The figure aggregates analyses across 3 Newton recording sessions.

## Data Sources

### Recording Sessions
- `data/rnn_export_Newton_08_13_2025.mat` - Session 1
- `data/rnn_export_Newton_08_14_2025.mat` - Session 2
- `data/rnn_export_Newton_08_15_2025.mat` - Session 3

### Trained Models
- `results/final_model/` - Full E/I model (08_14 session)
- `results/replication/` - Models for all 3 sessions
- `results/e_only_model/` - E-only models (to be trained)

---

## New Analyses Required

### 1. E-Only Model Training (Panel C)

**Purpose**: Train RNNs that only fit E (classic) neurons, leaving I neurons as unconstrained latent units. This reveals what inhibitory dynamics the network "needs" to fit the data.

**Method**:
- Same architecture: attention embedding, learnable h0, Dale's law enforced
- Same hyperparameters as full model
- Modify L_neuron to only include E neurons in target fitting
- I neurons in model are still present (Dale's law) but not constrained by data
- Extract learned I neuron PSTHs for comparison to recorded interneurons

**Implementation**:
- Modify `EIRNNLoss` to accept `fit_only_e=True` parameter
- When enabled, L_neuron only computes loss over E neuron indices
- Train for all 3 sessions

### 2. I→E Connectivity vs Factor Selectivity (Panel B)

**Purpose**: Show that inhibitory input to E neurons is uncorrelated with E neuron factor selectivity (evidence against structured lateral inhibition).

**Method**:
- For each E neuron, compute total inhibitory input: `sum(|W_EI[e, :]|)` (sum over all I neurons)
- For each E neuron, compute factor selectivity:
  - Reward selectivity: partial eta-squared for reward factor
  - Salience selectivity: partial eta-squared for salience factor
- Compute Pearson correlation between I input and each selectivity metric
- Aggregate across all 3 sessions (pool E neurons)

**Expected Result**: r ≈ 0, p > 0.05 (no correlation)

### 3. Input-Potent / Input-Null Subspace Decomposition (Panel D)

**Purpose**: Show that factor-related variance is predominantly input-driven rather than recurrence-driven.

**Method** (Kaufman et al., 2014):

1. **Define input-potent subspace**:
   - Compute trial-averaged population activity `X(t)` [n_neurons × n_timepoints]
   - Compute trial-averaged inputs `U(t)` [n_inputs × n_timepoints]
   - Regress activity onto inputs: `X = B @ U` → solve for `B` [n_neurons × n_inputs]
   - Input-potent subspace: column space of `B` (orthonormalize via QR)
   - This captures the dimensions of neural activity directly driven by inputs

2. **Define input-null subspace**:
   - Orthogonal complement to input-potent subspace
   - Represents recurrence-driven dynamics not explained by direct input

3. **Project population activity**:
   - Project `X(t)` onto input-potent basis → `X_potent(t)`
   - Project `X(t)` onto input-null basis → `X_null(t)`

4. **Compute factor-related variance**:
   - For each factor (reward, salience):
     - Compute trial-by-trial activity for high vs low factor levels
     - Compute variance explained by factor in each subspace
     - Proportion = var_subspace / var_total

**Expected Result**: Input-potent variance >> Input-null variance for both factors

### 4. Recorded Neuron Factor Selectivity (Panel E)

**Purpose**: Show that recorded interneurons lack factor selectivity compared to classic neurons.

**Method**:
- Compute ROC-based selectivity index for each neuron:
  - Reward selectivity: AUC for classifying reward vs no-reward trials
  - Salience selectivity: AUC for classifying high vs low salience trials
- Transform to selectivity index: `2 * (AUC - 0.5)` → range [-1, 1]
- Plot scatter: x = salience selectivity, y = reward selectivity
- Color/shape code by neuron type (E vs I)
- Aggregate across sessions

**Expected Result**: I neurons cluster near (0, 0); E neurons show distributed selectivity

---

## Data Aggregation Approach

### Cross-Session Pooling

For Panels B, D, and E, data is pooled across all 3 sessions:
- Maintain session identity for error bar computation (SEM across sessions)
- Neuron counts per session vary; weight equally or by n_neurons

### Session-Level Statistics

For Panel D (subspace analysis):
- Compute proportions separately per session
- Report mean ± SEM across sessions

### Normalization

- Factor selectivity metrics are comparable across sessions (unitless)
- I→E weights may differ in scale; normalize per session if needed

---

## Figure Generation Strategy

### Layout
```
+-------+-------+-------+
|   A   |   B   |   C   |
| Arch  | Conn  | Model |
+-------+-------+-------+
|   D   |   E   |       |
| Subs  |  ROC  |       |
+-------+-------+-------+
```

- **Dimensions**: 7" × 5" (180mm × 130mm)
- **DPI**: 300 for PNG, vector for PDF
- **Aspect ratio**: ~1.4:1

### Panel Specifications

#### Panel A: Architecture Schematic
- Manual illustration using matplotlib patches
- E neurons: filled circles (#2166AC)
- I neurons: open circles with orange border (#E66101)
- Arrows for connections with appropriate weights
- Labels: "Input", "E", "I", "Output", "Dale's Law"

#### Panel B: I→E Connectivity Scatter
- Scatter plot with small gray markers
- Regression line (thin gray)
- Statistics annotation (r, p)
- Consider two subpanels (B1: reward, B2: salience) or single combined plot

#### Panel C: Distribution Comparison
- Overlaid histograms or KDE plots
- Semi-transparent fills
- Model I units: purple (#762A83)
- Recorded interneurons: gray (#969696)

#### Panel D: Subspace Bar Chart
- Grouped bar chart
- Two groups (Reward, Salience)
- Two bars per group (Input-potent, Input-null)
- Error bars: SEM across sessions

#### Panel E: ROC Scatter
- Scatter plot
- E neurons: filled gray (#969696)
- I neurons: open orange (#E66101)
- Reference lines at selectivity = 0

---

## Panel-by-Panel Implementation Details

### Panel A Implementation

```python
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(figsize=(2, 2))

# Draw E neurons (filled blue)
for i, pos in enumerate(e_positions):
    circle = Circle(pos, radius=0.15, facecolor='#2166AC', edgecolor='black')
    ax.add_patch(circle)

# Draw I neurons (open orange)
for i, pos in enumerate(i_positions):
    circle = Circle(pos, radius=0.15, facecolor='white', edgecolor='#E66101', linewidth=2)
    ax.add_patch(circle)

# Draw connections (arrows)
# E→E, E→I, I→E, I→I
```

### Panel B Implementation

```python
# Aggregate data across sessions
all_e_selectivity_reward = []
all_e_selectivity_salience = []
all_i_input = []

for session in sessions:
    # Load model and data
    W_EI = model.get_weight_submatrices()['W_EI']  # [n_exc × n_inh]
    total_i_input = np.sum(np.abs(W_EI), axis=1)  # Sum over I neurons

    # Compute factor selectivity for E neurons
    selectivity = compute_factor_selectivity(data, 'reward')

    all_e_selectivity_reward.extend(selectivity[:n_exc])
    all_i_input.extend(total_i_input)

# Compute correlation
r, p = pearsonr(all_e_selectivity_reward, all_i_input)

# Plot scatter
ax.scatter(all_e_selectivity_reward, all_i_input, c='gray', s=10, alpha=0.5)
ax.plot(x_fit, y_fit, 'gray', linewidth=1)
ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.2f}', transform=ax.transAxes)
```

### Panel C Implementation

```python
# From E-only model: extract model I neuron PSTHs
model_i_psth = model_outputs[:, n_exc:]  # [n_time × n_inh]

# Compute factor selectivity for model I neurons
model_i_selectivity = compute_selectivity(model_i_psth, trial_labels)

# From data: recorded interneuron PSTHs
recorded_i_psth = target_data[:, inh_indices]
recorded_i_selectivity = compute_selectivity(recorded_i_psth, trial_labels)

# Plot overlaid histograms
ax.hist(np.abs(model_i_selectivity), bins=20, alpha=0.5, color='#762A83', label='Model I units')
ax.hist(np.abs(recorded_i_selectivity), bins=20, alpha=0.5, color='#969696', label='Recorded interneurons')
```

### Panel D Implementation

```python
def compute_subspace_variance(X, U, trial_labels, factor):
    """
    X: neural activity [n_trials × n_time × n_neurons]
    U: inputs [n_trials × n_time × n_inputs]
    trial_labels: dict with factor labels
    factor: 'reward' or 'salience'
    """
    # Compute input-potent subspace
    X_mean = X.mean(axis=0)  # [n_time × n_neurons]
    U_mean = U.mean(axis=0)  # [n_time × n_inputs]

    # Regression: X = B @ U
    B, _, _, _ = np.linalg.lstsq(U_mean.T, X_mean.T, rcond=None)
    B = B.T  # [n_neurons × n_inputs]

    # Orthonormalize columns of B via QR
    Q_potent, _ = np.linalg.qr(B)  # [n_neurons × k]

    # Input-null subspace: orthogonal complement
    n_neurons = X.shape[2]
    Q_null = null_space(B.T)  # [n_neurons × (n_neurons - k)]

    # Project activity
    X_flat = X.reshape(-1, n_neurons)  # [n_trials*n_time × n_neurons]
    X_potent = X_flat @ Q_potent  # Project onto potent
    X_null = X_flat @ Q_null  # Project onto null

    # Compute factor-related variance in each subspace
    labels = trial_labels[factor]
    var_potent = compute_factor_variance(X_potent.reshape(X.shape[0], X.shape[1], -1), labels)
    var_null = compute_factor_variance(X_null.reshape(X.shape[0], X.shape[1], -1), labels)

    return var_potent / (var_potent + var_null), var_null / (var_potent + var_null)
```

### Panel E Implementation

```python
def compute_roc_selectivity(firing_rates, trial_labels, factor):
    """
    Compute ROC-based selectivity for each neuron.
    Returns selectivity index: 2 * (AUC - 0.5), range [-1, 1]
    """
    selectivity = []
    for neuron in range(firing_rates.shape[1]):
        rates = firing_rates[:, neuron].mean(axis=1)  # Trial-averaged
        labels = trial_labels[factor]
        auc = roc_auc_score(labels, rates)
        selectivity.append(2 * (auc - 0.5))
    return np.array(selectivity)

# Plot
ax.scatter(e_salience, e_reward, c='#969696', s=20, label='E neurons')
ax.scatter(i_salience, i_reward, facecolor='none', edgecolor='#E66101', s=20, label='I neurons')
ax.axhline(0, linestyle='--', color='gray', linewidth=0.5)
ax.axvline(0, linestyle='--', color='gray', linewidth=0.5)
```

---

## File Organization

### Scripts
- `scripts/train_e_only_model.py` - Train E-only models for all sessions
- `scripts/compute_r01_analyses.py` - Compute all new analyses
- `scripts/generate_r01_figure.py` - Generate final figure

### Results
- `results/e_only_model/` - E-only model checkpoints and outputs
  - `session_08_13/`
  - `session_08_14/`
  - `session_08_15/`
- `results/r01_figure/` - Final figure outputs
  - `figure_rnn_analysis.png`
  - `figure_rnn_analysis.pdf`
  - `panel_data/` - CSV/NPY files for each panel

### Documentation
- `specs/r01_figure_report.md` - Methods, statistics, caption

---

## Implementation Timeline

1. **E-only model training** (~1-2 hours per session)
   - Modify loss function
   - Train for 3 sessions
   - Extract I neuron PSTHs

2. **Compute analyses** (~30 min)
   - Panel B: I→E connectivity correlations
   - Panel C: Model vs recorded I selectivity
   - Panel D: Subspace decomposition
   - Panel E: ROC selectivity scatter

3. **Generate figure** (~1 hour)
   - Create each panel
   - Assemble into final figure
   - Export PNG and PDF

4. **Documentation** (~30 min)
   - Write methods
   - Compile statistics
   - Draft caption

---

## Dependencies

### Python Packages
- numpy, scipy, matplotlib
- torch (for model loading/training)
- sklearn (for ROC computation)
- pandas (for data export)

### Data Requirements
- All 3 Newton session .mat files
- Trained full E/I models for each session
- (To be created) E-only models for each session

---

## Key References

- Kaufman MT et al. (2014) Nat Neurosci - Input-potent/null subspace decomposition
- Dale's principle - Enforced in EIRNN architecture
- Factor selectivity - Partial eta-squared from multiple regression

---

## Color Palette Summary

| Element | Color | Hex |
|---------|-------|-----|
| Goal-directed / Model I | Purple | #762A83 |
| Stimulus-driven | Teal/Green | #1B7837 |
| Neutral / Reference / Recorded data | Gray | #969696 |
| Input-potent / E neurons | Dark blue | #2166AC |
| Inhibitory neurons / Recorded I | Orange | #E66101 |
