# ANALYSIS_SPEC.md - Weight Structure Analysis

## Overview

This document specifies the analysis procedures for testing the central hypothesis: **Do SC putative interneurons provide factor-specific inhibition for competition resolution?**

After fitting the E-I RNN, we analyze the learned I→E weight structure to determine whether inhibitory connections are organized relative to factor selectivity.

---

## Central Analysis: I→E Weight Structure

### Hypothesis

**Null (External Loop Model)**: I→E weights are unstructured with respect to factor selectivity. Interneurons provide generic gain modulation; factor-specific competition occurs elsewhere (SNr, FEF, etc.).

**Alternative (Local Competition Model)**: I→E weights show factor-specific structure. Interneurons selectively inhibit neurons with similar/different factor preferences, implementing local winner-take-all dynamics.

### Weight Matrix Extraction

```python
def extract_IE_weights(model):
    """
    Extract weights from inhibitory to excitatory neurons.
    
    Returns:
        W_IE: [n_exc, n_inh] - Weight from I neuron j to E neuron i
    """
    W_rec = model.W_rec.detach().cpu().numpy()
    n_exc = model.n_exc
    
    # I→E submatrix: rows = E targets, cols = I sources
    # Note: I columns have negative values due to Dale's law
    W_IE = W_rec[:n_exc, n_exc:]  # [n_exc, n_inh]
    
    # Take absolute value for magnitude analysis
    # (sign is constrained to be negative by Dale's law)
    W_IE_mag = np.abs(W_IE)
    
    return W_IE, W_IE_mag
```

---

## Analysis 1: Factor Selectivity Correlation

**Question**: Are I neurons that project strongly to a given E neuron similar in their factor selectivity?

### Step 1: Compute Factor Selectivity for Each Neuron

```python
def compute_factor_selectivity(data, neuron_idx, factor='reward'):
    """
    Compute selectivity index for a given factor using ROC-AUC.
    
    Args:
        data: Exported data dictionary
        neuron_idx: Index of neuron
        factor: 'reward', 'salience', 'probability', or 'identity'
    
    Returns:
        selectivity: AUC value (0.5 = no selectivity, >0.5 = prefers high, <0.5 = prefers low)
    """
    firing_rates = data['firing_rates'][neuron_idx]  # [time, trials]
    
    # Get relevant trial labels
    if factor == 'reward':
        labels = data['trial_reward']  # 1 = high, 0 = low
    elif factor == 'salience':
        labels = data['trial_salience']
        # Filter to bullseye trials only
        valid = data['trial_identity'] == 3
        labels = labels[valid]
        firing_rates = firing_rates[:, valid]
    elif factor == 'probability':
        labels = data['trial_probability']
    elif factor == 'identity':
        labels = (data['trial_identity'] == 1).astype(int)  # face vs non-face
        valid = data['trial_identity'] <= 2  # exclude bullseye
        labels = labels[valid]
        firing_rates = firing_rates[:, valid]
    
    # Use visual epoch (50-250ms post-target)
    target_on_bin = find_event_bin(data, 'targetOn')
    epoch_start = target_on_bin + int(50 / data['bin_size_ms'])
    epoch_end = target_on_bin + int(250 / data['bin_size_ms'])
    
    # Mean firing rate in epoch
    epoch_fr = firing_rates[epoch_start:epoch_end, :].mean(axis=0)
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels, epoch_fr)
    except ValueError:
        auc = 0.5  # Undefined if only one class
    
    return auc


def compute_all_selectivities(data, neuron_indices, factors=['reward', 'salience', 'probability', 'identity']):
    """
    Compute selectivity matrix for all neurons and factors.
    
    Returns:
        selectivity: [n_neurons, n_factors] array
    """
    n_neurons = len(neuron_indices)
    n_factors = len(factors)
    selectivity = np.zeros((n_neurons, n_factors))
    
    for i, neuron_idx in enumerate(neuron_indices):
        for j, factor in enumerate(factors):
            selectivity[i, j] = compute_factor_selectivity(data, neuron_idx, factor)
    
    return selectivity, factors
```

### Step 2: Test Weight-Selectivity Relationship

```python
def analyze_weight_selectivity_correlation(W_IE_mag, E_selectivity, I_selectivity):
    """
    Test whether I→E weight magnitude correlates with selectivity similarity.
    
    Args:
        W_IE_mag: [n_exc, n_inh] - Absolute I→E weights
        E_selectivity: [n_exc, n_factors] - E neuron selectivities
        I_selectivity: [n_inh, n_factors] - I neuron selectivities
    
    Returns:
        results: dict with correlation statistics per factor
    """
    results = {}
    n_factors = E_selectivity.shape[1]
    
    for f in range(n_factors):
        # Selectivity vectors for this factor
        e_sel = E_selectivity[:, f]
        i_sel = I_selectivity[:, f]
        
        # For each E-I pair, compute selectivity similarity
        # (e.g., absolute difference or product)
        selectivity_similarity = np.outer(e_sel - 0.5, i_sel - 0.5)  # [n_exc, n_inh]
        # Positive = both prefer same direction
        # Negative = prefer opposite directions
        
        # Flatten for correlation
        weights_flat = W_IE_mag.flatten()
        similarity_flat = selectivity_similarity.flatten()
        
        # Pearson correlation
        r, p = pearsonr(weights_flat, similarity_flat)
        
        # Permutation test for significance
        p_perm = permutation_test_correlation(weights_flat, similarity_flat, n_permutations=1000)
        
        results[f'factor_{f}'] = {
            'correlation': r,
            'p_value': p,
            'p_permutation': p_perm
        }
    
    return results
```

### Step 3: Structured vs Random Weight Test

```python
def test_weight_structure(W_IE_mag, E_selectivity, I_selectivity, n_permutations=1000):
    """
    Test whether weights are more structured than expected by chance.
    
    Null hypothesis: I→E weights are independent of factor selectivity.
    """
    # Observed statistic: variance explained by selectivity
    observed_r2 = compute_weight_variance_explained(W_IE_mag, E_selectivity, I_selectivity)
    
    # Permutation distribution: shuffle neuron identities
    permuted_r2 = []
    for _ in range(n_permutations):
        # Shuffle E neuron labels
        E_sel_shuffled = E_selectivity[np.random.permutation(len(E_selectivity))]
        r2 = compute_weight_variance_explained(W_IE_mag, E_sel_shuffled, I_selectivity)
        permuted_r2.append(r2)
    
    permuted_r2 = np.array(permuted_r2)
    
    # P-value: proportion of permuted >= observed
    p_value = (permuted_r2 >= observed_r2).mean()
    
    return {
        'observed_r2': observed_r2,
        'permuted_r2_mean': permuted_r2.mean(),
        'permuted_r2_std': permuted_r2.std(),
        'p_value': p_value,
        'structured': p_value < 0.05
    }
```

---

## Analysis 2: Factor Decoding With/Without I→E Weights

**Question**: Do I→E connections contribute to factor discrimination in E neuron activity?

### Approach

Compare factor decoding accuracy from E neuron activity under two conditions:
1. **Full model**: Normal forward pass
2. **Ablated model**: Zero out I→E weights

```python
def compare_decoding_with_ablation(model, data, factor='reward'):
    """
    Compare factor decoding with and without I→E connections.
    """
    # Get inputs and run full model
    inputs = construct_inputs(data)
    
    with torch.no_grad():
        # Full model
        rates_full, _ = model(inputs)
        E_rates_full = rates_full[:, :, :model.n_exc]
        
        # Ablated model: zero I→E weights
        model_ablated = copy_model_with_ablation(model)
        rates_ablated, _ = model_ablated(inputs)
        E_rates_ablated = rates_ablated[:, :, :model.n_exc]
    
    # Decode factor from E activity
    labels = get_factor_labels(data, factor)
    
    accuracy_full = decode_factor(E_rates_full, labels)
    accuracy_ablated = decode_factor(E_rates_ablated, labels)
    
    return {
        'accuracy_full': accuracy_full,
        'accuracy_ablated': accuracy_ablated,
        'accuracy_diff': accuracy_full - accuracy_ablated,
        'contribution': (accuracy_full - accuracy_ablated) / accuracy_full
    }


def copy_model_with_ablation(model):
    """Create model copy with I→E weights zeroed."""
    import copy
    model_ablated = copy.deepcopy(model)
    
    with torch.no_grad():
        # Zero out I→E weights (first n_exc rows, last n_inh columns)
        model_ablated.W_rec_raw[:model.n_exc, model.n_exc:] = 0
    
    return model_ablated
```

### Interpretation

| Result | Interpretation |
|--------|----------------|
| accuracy_diff ≈ 0 | I→E weights don't contribute to factor encoding |
| accuracy_diff > 0 | I→E weights enhance factor discrimination |
| accuracy_diff < 0 | I→E weights suppress factor discrimination (unexpected) |

---

## Analysis 3: Comparison Across Model Variants

### Model Variants

| Variant | Description | Purpose |
|---------|-------------|---------|
| **A** | Full constraints, E/I assigned by physiology | Primary test |
| **B** | Unconstrained E/I (learn from data) | Does functional ≈ physiological? |
| **C** | Classic neurons only (invent I) | What would optimal I look like? |

### Variant B Analysis: Functional vs Physiological Classification

```python
def analyze_functional_classification(model_B, data):
    """
    In unconstrained model, which neurons end up functionally inhibitory?
    Compare to physiological classification.
    """
    W_rec = model_B.W_rec.detach().cpu().numpy()
    
    # Functional classification: net output sign
    # If sum of outgoing weights is negative, functionally inhibitory
    net_output = W_rec.sum(axis=0)  # Sum over target neurons
    functional_I = net_output < 0
    
    # Physiological classification from data
    physiological_I = data['neuron_type'] == 2
    
    # Agreement
    agreement = (functional_I == physiological_I).mean()
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(physiological_I, functional_I)
    
    return {
        'agreement': agreement,
        'confusion_matrix': cm,
        'n_physio_I_functional_I': cm[1, 1],  # True positives
        'n_physio_I_functional_E': cm[1, 0],  # Putative INTs that act excitatory
    }
```

### Variant C Analysis: Invented vs Actual Interneurons

```python
def compare_invented_vs_actual_interneurons(model_C, data):
    """
    Compare response profiles of invented I units to actual putative interneurons.
    """
    # Get activity from model C (which invents its own I units)
    rates_C = run_model(model_C, data)
    invented_I_rates = rates_C[:, :, model_C.n_exc:]  # Hidden I units
    
    # Get actual putative interneuron activity from data
    actual_I_idx = np.where(data['neuron_type'] == 2)[0]
    actual_I_rates = data['firing_rates'][actual_I_idx]
    
    # Compare response profiles
    # 1. Trial-averaged activity correlation
    invented_psth = invented_I_rates.mean(axis=0)  # [time, n_invented_I]
    actual_psth = actual_I_rates.mean(axis=2)  # [n_actual_I, time]
    
    # Cross-correlation matrix
    correlations = np.corrcoef(
        np.vstack([invented_psth.T, actual_psth])
    )
    
    # 2. Factor selectivity comparison
    invented_selectivity = compute_all_selectivities_from_rates(invented_I_rates)
    actual_selectivity = compute_all_selectivities(data, actual_I_idx)
    
    return {
        'psth_correlation_matrix': correlations,
        'invented_selectivity': invented_selectivity,
        'actual_selectivity': actual_selectivity,
        'selectivity_similarity': compare_selectivity_distributions(
            invented_selectivity, actual_selectivity
        )
    }
```

---

## Visualization

### 1. Weight Matrix Heatmap

```python
def plot_weight_matrix(W_IE, E_selectivity, I_selectivity, factor_idx=0):
    """
    Plot I→E weights sorted by factor selectivity.
    """
    # Sort E neurons by selectivity
    e_order = np.argsort(E_selectivity[:, factor_idx])
    # Sort I neurons by selectivity
    i_order = np.argsort(I_selectivity[:, factor_idx])
    
    W_sorted = W_IE[e_order][:, i_order]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(W_sorted, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.abs(W_sorted).max(), vmax=np.abs(W_sorted).max())
    ax.set_xlabel('I neurons (sorted by selectivity)')
    ax.set_ylabel('E neurons (sorted by selectivity)')
    ax.set_title(f'I→E Weights Sorted by Factor {factor_idx} Selectivity')
    plt.colorbar(im)
    
    return fig
```

### 2. Selectivity vs Weight Scatter

```python
def plot_selectivity_weight_scatter(W_IE_mag, E_selectivity, I_selectivity, factor_idx=0):
    """
    Scatter plot of weight magnitude vs selectivity similarity.
    """
    e_sel = E_selectivity[:, factor_idx] - 0.5
    i_sel = I_selectivity[:, factor_idx] - 0.5
    
    similarity = np.outer(e_sel, i_sel).flatten()
    weights = W_IE_mag.flatten()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(similarity, weights, alpha=0.3, s=10)
    ax.set_xlabel('Selectivity Similarity (E × I)')
    ax.set_ylabel('|I→E Weight|')
    
    # Add regression line
    slope, intercept, r, p, se = linregress(similarity, weights)
    x_line = np.linspace(similarity.min(), similarity.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', 
            label=f'r={r:.3f}, p={p:.3e}')
    ax.legend()
    
    return fig
```

### 3. Decoding Comparison Bar Plot

```python
def plot_decoding_comparison(results_dict):
    """
    Bar plot comparing factor decoding with/without I→E weights.
    """
    factors = list(results_dict.keys())
    full = [results_dict[f]['accuracy_full'] for f in factors]
    ablated = [results_dict[f]['accuracy_ablated'] for f in factors]
    
    x = np.arange(len(factors))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, full, width, label='Full Model')
    ax.bar(x + width/2, ablated, width, label='I→E Ablated')
    ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylabel('Decoding Accuracy')
    ax.legend()
    
    return fig
```

---

## Expected Outcomes

### If Hypothesis Supported (External Loop Model)

1. **Weight-selectivity correlation ≈ 0**: I→E weights not organized by factor preference
2. **Ablation effect ≈ 0**: Factor decoding unchanged without I→E connections
3. **Variant B**: Functional classification ≠ physiological (putative INTs don't act inhibitory in learned circuit)
4. **Variant C**: Invented I units don't resemble actual putative interneurons

### If Alternative (Local Competition Model)

1. **Significant weight-selectivity correlation**: Strong I→E connections between similarly/dissimilarly tuned neurons
2. **Ablation degrades decoding**: I→E connections necessary for factor discrimination
3. **Variant B**: Good agreement between functional and physiological classification
4. **Variant C**: Invented I units resemble actual putative interneurons

---

## Reporting

### Minimum Results to Report

1. Weight structure statistics (correlation, permutation test p-value)
2. Decoding accuracy with/without ablation per factor
3. Weight matrix heatmaps (sorted by selectivity)
4. Model fit quality (PSTH correlation, trial-matching loss)

### For R01 Proposal

Focus on:
- Clear statement of prediction before showing data
- Visualization of weight structure (or lack thereof)
- Effect size, not just significance
- Comparison to what local competition would predict
