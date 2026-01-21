# TRAINING_SPEC.md - Loss Functions and Optimization

## Overview

This document specifies the loss functions, optimization procedure, and training pipeline for fitting the E-I RNN to SC neural recordings. The approach follows Sourmpis et al. (2026) with adaptations for single-area recordings.

## Loss Function

### Total Loss

```
L = L_neuron + L_trial + λ_reg · L_reg
```

All components are weighted using the parameter-free multi-task reweighting from Défossez et al. (2023), which normalizes gradient magnitudes across loss terms.

---

### L_neuron: Trial-Averaged Activity Loss

**Purpose**: Ensure each RNN unit matches the PSTH of its corresponding recorded neuron.

**Formulation**:

```python
def compute_L_neuron(model_rates, target_rates, time_mask=None):
    """
    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_neurons] - Recorded firing rates
        time_mask: [batch, time] - Valid timesteps (1) vs padding (0)
    
    Returns:
        L_neuron: scalar loss
    """
    # Trial-average
    model_psth = model_rates.mean(dim=0)  # [time, n_neurons]
    target_psth = target_rates.mean(dim=0)  # [time, n_neurons]
    
    # Temporal smoothing (8ms Gaussian kernel, as in Sourmpis)
    kernel_size = max(1, int(8 / bin_size_ms))  # bins
    model_psth = smooth_temporal(model_psth, kernel_size)
    target_psth = smooth_temporal(target_psth, kernel_size)
    
    # Z-score normalize per neuron (across time)
    model_psth_norm = (model_psth - model_psth.mean(dim=0)) / (model_psth.std(dim=0) + 1e-8)
    target_psth_norm = (target_psth - target_psth.mean(dim=0)) / (target_psth.std(dim=0) + 1e-8)
    
    # MSE across time and neurons
    L_neuron = ((model_psth_norm - target_psth_norm) ** 2).mean()
    
    return L_neuron
```

**Key Details**:
- Temporal smoothing with 8ms kernel (following Sourmpis)
- Z-score normalization per neuron prevents high-FR neurons from dominating
- Applied only to recorded neurons (not hidden units)

---

### L_trial: Trial-Matching Loss

**Purpose**: Capture single-trial variability, not just averages. This is critical—fitting only PSTHs produces networks that lack realistic trial-to-trial dynamics.

**Formulation** (Sourmpis et al. 2023, 2026):

```python
def compute_L_trial(model_rates, target_rates, time_mask=None):
    """
    Compute trial-matching loss using optimal transport-inspired matching.
    
    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_neurons] - Recorded firing rates
    
    Returns:
        L_trial: scalar loss
    """
    batch_size = model_rates.shape[0]
    
    # Population-average activity per trial (32ms smoothing)
    kernel_size = max(1, int(32 / bin_size_ms))
    model_pop = smooth_temporal(model_rates.mean(dim=2), kernel_size)  # [batch, time]
    target_pop = smooth_temporal(target_rates.mean(dim=2), kernel_size)  # [batch, time]
    
    # Z-score normalize across trials (per timepoint)
    model_pop_norm = (model_pop - model_pop.mean(dim=0)) / (model_pop.std(dim=0) + 1e-8)
    target_pop_norm = (target_pop - target_pop.mean(dim=0)) / (target_pop.std(dim=0) + 1e-8)
    
    # Compute pairwise distances: [batch_model, batch_target]
    # Each model trial matched to closest target trial
    distances = torch.cdist(model_pop_norm, target_pop_norm, p=2)  # Euclidean
    
    # Hungarian algorithm for optimal assignment (or greedy approximation)
    # For efficiency, use greedy: each model trial → closest unmatched target
    min_distances, assignments = greedy_match(distances)
    
    # Loss is mean matched distance
    L_trial = min_distances.mean()
    
    return L_trial


def greedy_match(distances):
    """Greedy approximation to optimal assignment."""
    batch_size = distances.shape[0]
    matched_dists = []
    available = set(range(batch_size))
    
    for i in range(batch_size):
        # Find closest available target for model trial i
        dists_i = distances[i, list(available)]
        idx_in_available = dists_i.argmin()
        target_idx = list(available)[idx_in_available]
        matched_dists.append(distances[i, target_idx])
        available.remove(target_idx)
    
    return torch.stack(matched_dists), None
```

**Key Details**:
- Population-averaged activity captures collective network state
- 32ms smoothing kernel (coarser than L_neuron)
- Trial matching ensures model produces realistic variability distribution
- Greedy matching is O(n²); use Hungarian algorithm for exact solution if needed

---

### L_reg: Regularization Loss

**Purpose**: Prevent overfitting, encourage sparse/interpretable solutions.

```python
def compute_L_reg(model, lambda_l2=1e-4, lambda_sparse=0.0):
    """
    Weight regularization.
    
    Args:
        model: EIRNN model
        lambda_l2: L2 regularization strength
        lambda_sparse: Sparsity penalty (optional, Sourmpis found not critical)
    
    Returns:
        L_reg: scalar loss
    """
    W_rec = model.W_rec  # With Dale's law enforced
    W_in = model.W_in
    
    # L2 regularization
    L_l2 = (W_rec ** 2).mean() + (W_in ** 2).mean()
    
    # Optional: L1/2 sparsity (Sourmpis Eq. 10)
    if lambda_sparse > 0:
        L_sparse = (torch.abs(W_rec) ** 0.5).mean()
    else:
        L_sparse = 0.0
    
    L_reg = lambda_l2 * L_l2 + lambda_sparse * L_sparse
    
    return L_reg
```

---

## Optimization

### Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### Learning Rate Schedule

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=50,
    min_lr=1e-5
)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Training Loop

```python
def train_epoch(model, data_loader, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss = 0
    loss_components = {'L_neuron': 0, 'L_trial': 0, 'L_reg': 0}
    
    for batch in data_loader:
        inputs = batch['inputs'].to(device)       # [batch, time, n_inputs]
        targets = batch['firing_rates'].to(device) # [batch, time, n_neurons]
        mask = batch.get('mask', None)            # [batch, time]
        
        optimizer.zero_grad()
        
        # Forward pass
        model_rates, model_outputs = model(inputs)
        
        # Compute losses (only for recorded neurons)
        n_recorded = targets.shape[2]
        model_rates_recorded = model_rates[:, :, :n_recorded]
        
        L_neuron = compute_L_neuron(model_rates_recorded, targets, mask)
        L_trial = compute_L_trial(model_rates_recorded, targets, mask)
        L_reg = compute_L_reg(model)
        
        # Total loss with gradient normalization
        loss = normalize_and_combine([L_neuron, L_trial, L_reg])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        loss_components['L_neuron'] += L_neuron.item()
        loss_components['L_trial'] += L_trial.item()
        loss_components['L_reg'] += L_reg.item()
    
    return total_loss, loss_components


def normalize_and_combine(losses):
    """
    Parameter-free multi-task loss balancing (Défossez et al. 2023).
    Normalizes gradients to have comparable magnitudes.
    """
    # Simple version: weight by inverse of loss magnitude
    weights = [1.0 / (l.detach() + 1e-8) for l in losses]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    return sum(w * l for w, l in zip(weights, losses))
```

---

## Training Configuration

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | Full session | All trials in one batch for trial-matching |
| Learning rate | 1e-3 | Adam default |
| Epochs | 1000-2000 | Early stopping based on validation |
| λ_reg (L2) | 1e-4 | Standard regularization |
| λ_sparse | 0 | Sourmpis found not critical |
| Gradient clip | 1.0 | Prevent exploding gradients |

### Data Splitting

```python
# 80/20 train/validation split
n_trials = data['n_trials']
n_train = int(0.8 * n_trials)

# Stratified by condition if possible
train_idx, val_idx = stratified_split(
    trial_labels=data['trial_reward'],  # Or multi-factor stratification
    train_size=0.8
)
```

### Early Stopping

```python
patience = 100  # epochs
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        save_checkpoint(model, 'best_model.pt')
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## Validation Metrics

Beyond loss, track these during training:

### 1. PSTH Correlation

```python
def psth_correlation(model_rates, target_rates):
    """Pearson correlation of trial-averaged activity per neuron."""
    model_psth = model_rates.mean(dim=0)  # [time, neurons]
    target_psth = target_rates.mean(dim=0)
    
    correlations = []
    for i in range(model_psth.shape[1]):
        r = pearsonr(model_psth[:, i], target_psth[:, i])[0]
        correlations.append(r)
    
    return np.mean(correlations)
```

### 2. Trial-Type Accuracy

```python
def trial_type_accuracy(model, data):
    """Can we decode trial type (e.g., high/low reward) from model activity?"""
    # Compare to data decoding accuracy
    pass
```

### 3. Weight Constraint Verification

```python
def verify_constraints(model):
    """Ensure Dale's law and other constraints are maintained."""
    W_rec = model.W_rec.detach().cpu().numpy()
    n_exc = model.n_exc
    
    # Check E columns are positive
    assert (W_rec[:, :n_exc] >= 0).all(), "Dale's law violated: E weights negative"
    
    # Check I columns are negative
    assert (W_rec[:, n_exc:] <= 0).all(), "Dale's law violated: I weights positive"
    
    # Check diagonal is zero
    assert np.allclose(np.diag(W_rec), 0), "Self-connections exist"
    
    return True
```

---

## Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyperparameters': {
            'n_exc': model.n_exc,
            'n_inh': model.n_inh,
            'tau': model.tau,
            'dt': model.dt,
            # ... other hyperparameters
        }
    }, path)


def load_checkpoint(path, model_class):
    """Load trained model."""
    checkpoint = torch.load(path)
    
    # Reconstruct model
    hp = checkpoint['hyperparameters']
    model = model_class(**hp)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint
```

---

## Troubleshooting

### Loss Not Decreasing
1. Check learning rate (try 1e-4)
2. Verify data normalization
3. Check for NaN in inputs
4. Reduce batch size if memory-limited

### Exploding Gradients
1. Reduce learning rate
2. Increase gradient clipping threshold
3. Check spectral radius of W_rec initialization (<1.0)

### Poor Validation Performance
1. Increase regularization (λ_reg)
2. Reduce model capacity (fewer hidden units)
3. Check for data leakage in train/val split

### Dale's Law Violated After Training
1. Ensure using `model.W_rec` property, not `model.W_rec_raw`
2. Check sign_mask is properly registered as buffer
3. Verify no in-place operations on constrained weights

---

## References

- Sourmpis et al. (2023). Trial matching: capturing variability with data-constrained spiking neural networks. NeurIPS.
- Sourmpis et al. (2026). Biologically informed cortical models predict optogenetic perturbations. eLife.
- Défossez et al. (2023). Parameter-free multi-task loss balancing.
