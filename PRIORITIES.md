# PRIORITIES.md - RNN Training Fix Roadmap

## Executive Summary

The E-I RNN is collapsing to a degenerate low-activity solution instead of learning SC neural dynamics. This document provides a prioritized roadmap for fixes. **Read the entire document before implementing**—the issues are interconnected.

## Diagnosis

### Root Cause: Scale Mismatch + Loss Function Interaction

1. **Scale Problem**: softplus activation naturally outputs ~1 sp/s; target data is ~11 sp/s mean
2. **Loss Instability**: Z-score normalization explodes when model has low temporal variance
3. **L_trial Dominance**: Trial-matching loss plateaus because flat model outputs are indistinguishable
4. **Weak Input Drive**: External inputs don't sufficiently modulate network activity

### Evidence from Training Curves

- 1e20 spike in L_neuron = division by near-zero std during normalization
- L_trial ~4.8 while L_neuron ~0.5 = trial loss dominates but provides no gradient
- PSTH correlation stuck at ~0.03 = model isn't capturing temporal dynamics
- Model outputs ~1 sp/s vs target ~11 sp/s = 10x scale mismatch

---

## Implementation Priorities

### Priority 0: Diagnostic Baseline (Do First)

Before any fixes, establish what's actually happening. Add this diagnostic script:

```python
# scripts/diagnose_model.py
"""Diagnostic script to understand model behavior before fixes."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.data_loader import load_session, train_val_split


def diagnose(data_path: str):
    """Run diagnostics on model behavior."""
    
    # Load data
    dataset = load_session(data_path)
    all_data = dataset.get_all_trials()
    
    # Data statistics
    targets = all_data['targets']
    print("=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"Target shape: {targets.shape}")
    print(f"Target mean: {targets.mean():.2f} sp/s")
    print(f"Target std: {targets.std():.2f} sp/s")
    print(f"Target max: {targets.max():.2f} sp/s")
    print(f"Target min: {targets.min():.2f} sp/s")
    print(f"Target 99th percentile: {np.percentile(targets.numpy(), 99):.2f} sp/s")
    
    # Check for outliers
    outlier_threshold = 200  # sp/s
    n_outliers = (targets > outlier_threshold).sum().item()
    print(f"Values > {outlier_threshold} sp/s: {n_outliers} ({100*n_outliers/targets.numel():.3f}%)")
    
    # PSTH statistics (trial-averaged)
    psth = targets.mean(dim=0)  # [time, neurons]
    print(f"\nPSTH temporal std (mean across neurons): {psth.std(dim=0).mean():.3f}")
    print(f"PSTH neuron std (mean across time): {psth.std(dim=1).mean():.3f}")
    
    # Create model
    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()
    
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        device='cpu'
    )
    
    print("\n" + "=" * 60)
    print("MODEL BEHAVIOR (untrained)")
    print("=" * 60)
    
    with torch.no_grad():
        inputs = all_data['inputs'][:10]
        
        # Test with actual inputs
        rates_actual, _ = model(inputs)
        print(f"Rates with actual input: {rates_actual.mean():.3f} ± {rates_actual.std():.3f} sp/s")
        
        # Test with zero inputs
        zero_inputs = torch.zeros_like(inputs)
        rates_zero, _ = model(zero_inputs)
        print(f"Rates with zero input: {rates_zero.mean():.3f} ± {rates_zero.std():.3f} sp/s")
        
        # Test with 10x inputs
        rates_10x, _ = model(inputs * 10)
        print(f"Rates with 10x input: {rates_10x.mean():.3f} ± {rates_10x.std():.3f} sp/s")
        
        # Input modulation ratio
        modulation = (rates_actual.mean() - rates_zero.mean()) / (rates_zero.mean() + 1e-8)
        print(f"\nInput modulation ratio: {modulation:.2%}")
        
        # Temporal dynamics check
        rates_temporal_std = rates_actual.std(dim=1).mean()  # std across time
        target_temporal_std = targets[:10].std(dim=1).mean()
        print(f"\nModel temporal std: {rates_temporal_std:.3f}")
        print(f"Target temporal std: {target_temporal_std:.3f}")
        print(f"Ratio: {rates_temporal_std/target_temporal_std:.2%}")
    
    # Weight statistics
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)
    W_rec = model.W_rec.detach().numpy()
    W_in = model.W_in.detach().numpy()
    
    print(f"W_rec: shape {W_rec.shape}, range [{W_rec.min():.4f}, {W_rec.max():.4f}]")
    print(f"W_in: shape {W_in.shape}, range [{W_in.min():.4f}, {W_in.max():.4f}]")
    print(f"W_rec spectral radius: {np.max(np.abs(np.linalg.eigvals(W_rec))):.4f}")
    
    # Input signal statistics
    print("\n" + "=" * 60)
    print("INPUT SIGNAL STATISTICS")
    print("=" * 60)
    inputs_all = all_data['inputs']
    print(f"Inputs shape: {inputs_all.shape}")
    print(f"Inputs mean: {inputs_all.mean():.4f}")
    print(f"Inputs std: {inputs_all.std():.4f}")
    print(f"Inputs range: [{inputs_all.min():.4f}, {inputs_all.max():.4f}]")
    
    # Per-channel statistics
    for i in range(min(5, inputs_all.shape[2])):
        ch = inputs_all[:, :, i]
        print(f"  Channel {i}: mean={ch.mean():.4f}, std={ch.std():.4f}, nonzero={(ch != 0).float().mean():.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .mat file")
    args = parser.parse_args()
    
    diagnose(args.data)
```

**Run this first**: `python scripts/diagnose_model.py --data data/rnn_export_Newton_08_14_2025_SC.mat`

This will reveal:
- Whether inputs actually modulate the network
- The exact scale mismatch
- Whether there are data outliers to handle
- The spectral radius and weight magnitudes

---

### Priority 1: Fix the Architecture (Scale-Aware Firing Rates)

**File**: `src/model.py`

**Problem**: softplus outputs ~0.7-1 sp/s at initialization; needs to output ~11 sp/s.

**Solution**: Add learnable scale and baseline parameters.

```python
# In EIRNN.__init__, after existing parameter definitions, add:

# Learnable rate scaling (initialize near target mean firing rate)
self.rate_scale = nn.Parameter(torch.tensor(10.0))
self.rate_baseline = nn.Parameter(torch.tensor(0.5))

# In EIRNN.forward(), replace the rate computation:
# OLD: r = torch.nn.functional.softplus(x)
# NEW:
r_raw = torch.nn.functional.softplus(x)
r = r_raw * self.rate_scale + self.rate_baseline
r = torch.clamp(r, min=0.0)  # Ensure non-negative after baseline shift
```

**Also add** stronger input initialization:

```python
# In EIRNN._initialize_weights(), replace W_in initialization:
# OLD: nn.init.uniform_(self.W_in, -1.0, 1.0)
# NEW: Stronger input drive, bias toward positive (excitatory drive)
nn.init.uniform_(self.W_in, -0.5, 2.0)
```

---

### Priority 2: Fix the Loss Function (Robust PSTH Loss)

**File**: `src/losses.py`

**Problem**: Z-score normalization explodes when model has low variance.

**Solution**: Replace with correlation-based loss + explicit scale matching.

```python
def compute_L_neuron(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    normalize: bool = True  # Keep parameter for API compatibility, but ignore
) -> torch.Tensor:
    """
    Compute neuron-wise PSTH loss using correlation + scale matching.
    
    This is more robust than z-score MSE when model variance is low.
    """
    # Select recorded neurons
    n_recorded = target_rates.shape[2]
    if recorded_indices is not None:
        model_rates = model_rates[:, :, recorded_indices]
    else:
        model_rates = model_rates[:, :, :n_recorded]
    
    # Apply mask
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        model_rates = model_rates * mask_expanded
        target_rates = target_rates * mask_expanded
    
    # Trial-average to get PSTH
    model_psth = model_rates.mean(dim=0)  # [time, neurons]
    target_psth = target_rates.mean(dim=0)
    
    # Temporal smoothing
    kernel_size = max(1, int(smooth_ms / bin_size_ms))
    model_psth = smooth_temporal(model_psth.unsqueeze(0), kernel_size, dim=1).squeeze(0)
    target_psth = smooth_temporal(target_psth.unsqueeze(0), kernel_size, dim=1).squeeze(0)
    
    # === CORRELATION LOSS (shape matching) ===
    # Center the signals
    model_centered = model_psth - model_psth.mean(dim=0, keepdim=True)
    target_centered = target_psth - target_psth.mean(dim=0, keepdim=True)
    
    # Compute correlation per neuron
    numerator = (model_centered * target_centered).sum(dim=0)
    model_norm = (model_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-6)
    target_norm = (target_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-6)
    
    correlation = numerator / (model_norm * target_norm)
    
    # Loss: minimize (1 - correlation), averaged across neurons
    # Clamp correlation to [-1, 1] for numerical stability
    correlation = torch.clamp(correlation, -1.0, 1.0)
    L_corr = (1.0 - correlation).mean()
    
    # === SCALE LOSS (magnitude matching) ===
    # Penalize difference in mean firing rates
    model_mean = model_psth.mean()
    target_mean = target_psth.mean()
    target_var = target_psth.var().clamp(min=1e-6)
    
    L_scale = ((model_mean - target_mean) ** 2) / target_var
    
    # === VARIANCE LOSS (dynamics magnitude) ===
    # Encourage model to have similar temporal variance
    model_var = model_psth.var(dim=0).mean()
    target_var_per_neuron = target_psth.var(dim=0).mean()
    
    L_var = ((model_var - target_var_per_neuron) ** 2) / (target_var_per_neuron ** 2 + 1e-6)
    
    # Combine: prioritize correlation, but include scale and variance
    L_neuron = L_corr + 0.1 * L_scale + 0.05 * L_var
    
    return L_neuron
```

---

### Priority 3: Data Preprocessing (Handle Outliers)

**File**: `src/data_loader.py`

**Problem**: 720 sp/s max suggests outliers that destabilize training.

**Solution**: Add outlier clipping in the data loader.

```python
# In load_session() or the dataset class, after loading firing rates:

def preprocess_firing_rates(firing_rates: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Preprocess firing rates: clip outliers, ensure non-negative.
    
    Args:
        firing_rates: [trials, time, neurons] array
        clip_percentile: Percentile for outlier clipping
    
    Returns:
        Preprocessed firing rates
    """
    # Compute clip threshold
    clip_threshold = np.percentile(firing_rates, clip_percentile)
    
    # Report clipping
    n_clipped = (firing_rates > clip_threshold).sum()
    if n_clipped > 0:
        print(f"Clipping {n_clipped} values ({100*n_clipped/firing_rates.size:.3f}%) above {clip_threshold:.1f} sp/s")
    
    # Clip
    firing_rates = np.clip(firing_rates, 0, clip_threshold)
    
    return firing_rates
```

Call this in the data loading pipeline before converting to tensors.

---

### Priority 4: Curriculum Learning (Staged Training)

**File**: `scripts/train_model.py`

**Problem**: Model can't learn PSTH shape and trial structure simultaneously from scratch.

**Solution**: Two-stage training—first match PSTHs, then add trial matching.

```python
# Modify the train() function to support curriculum learning:

def train(
    data_path: str,
    output_dir: str,
    max_epochs: int = 1000,
    warmup_epochs: int = 200,  # NEW: epochs with L_neuron only
    patience: int = 100,
    lr: float = 1e-3,
    lambda_neuron: float = 1.0,
    lambda_trial: float = 1.0,
    lambda_reg: float = 1e-4,
    enforce_ratio: bool = True,
    device: str = 'cpu',
    seed: int = 42
):
    """
    Training with curriculum learning.
    
    Stage 1 (epochs 0 to warmup_epochs): L_neuron only (PSTH matching)
    Stage 2 (epochs warmup_epochs to end): L_neuron + L_trial (add trial structure)
    """
    
    # ... existing setup code ...
    
    for epoch in pbar:
        # Curriculum: ramp up L_trial after warmup
        if epoch < warmup_epochs:
            current_lambda_trial = 0.0
            stage = "warmup"
        else:
            # Linear ramp from 0 to lambda_trial over 100 epochs
            ramp_progress = min(1.0, (epoch - warmup_epochs) / 100.0)
            current_lambda_trial = lambda_trial * ramp_progress
            stage = "full"
        
        # Update loss function weights
        loss_fn.lambda_trial = current_lambda_trial
        
        # ... rest of training loop ...
        
        # Update progress bar with stage info
        pbar.set_postfix({
            'stage': stage,
            'λ_trial': f"{current_lambda_trial:.2f}",
            'train': f"{train_metrics['total']:.4f}",
            'val': f"{val_metrics['total']:.4f}",
            'corr': f"{val_psth_corr:.3f}",
        })
```

---

### Priority 5: Sanity Check—Unconstrained RNN

**File**: `src/model.py`

**Problem**: Need to verify the architecture can learn at all before debugging E-I constraints.

**Solution**: Add bypass option for Dale's law.

```python
# In EIRNN.__init__, add parameter:
def __init__(
    self,
    n_exc: int,
    n_inh: int,
    n_inputs: int,
    n_outputs: int = 2,
    tau: float = 50.0,
    dt: float = 25.0,
    noise_scale: float = 0.01,
    spectral_radius: float = 0.9,
    bypass_dale: bool = False,  # NEW: for debugging
    device: str = 'cpu'
):
    # ... existing code ...
    self.bypass_dale = bypass_dale

# Modify the W_rec property:
@property
def W_rec(self) -> torch.Tensor:
    """Recurrent weights with Dale's law enforced (unless bypassed)."""
    if self.bypass_dale:
        # No constraints—for debugging only
        W = self.W_rec_raw * self.diag_mask
    else:
        # Apply Dale's law: |W| * sign_mask * diag_mask
        W = torch.abs(self.W_rec_raw) * self.sign_mask * self.diag_mask
    return W
```

**Test procedure**:
1. Train with `bypass_dale=True`
2. If PSTH correlation reaches >0.3, the data is learnable—problem is with E-I constraints
3. If still fails, problem is elsewhere (loss function, data, hyperparameters)

---

## Testing Protocol

After implementing all fixes, run this sequence:

### Test 1: Diagnostic
```bash
python scripts/diagnose_model.py --data data/rnn_export_Newton_08_14_2025_SC.mat
```
Verify that input modulation ratio is now significant (>50%).

### Test 2: Quick Training Run
```bash
python scripts/train_model.py \
    --data data/rnn_export_Newton_08_14_2025_SC.mat \
    --output results/test_fixed/ \
    --max_epochs 100 \
    --warmup_epochs 50
```

**Success criteria for 100 epochs**:
- PSTH correlation > 0.15 (should be improving steadily)
- Model mean firing rate within 2x of target
- No loss explosions (no values > 1e6)

### Test 3: Full Training
```bash
python scripts/train_model.py \
    --data data/rnn_export_Newton_08_14_2025_SC.mat \
    --output results/Newton_fixed/ \
    --max_epochs 500 \
    --warmup_epochs 200 \
    --patience 100
```

**Success criteria**:
- PSTH correlation > 0.3 (ideally > 0.5)
- Validation loss decreasing or stable
- Training curves show clear learning (not flat)

### Test 4: Unconstrained Sanity Check (if Tests 2-3 fail)
```bash
python scripts/train_model.py \
    --data data/rnn_export_Newton_08_14_2025_SC.mat \
    --output results/test_unconstrained/ \
    --max_epochs 200 \
    --bypass_dale
```

If unconstrained works but constrained doesn't, E-I ratio may need adjustment.

---

## Success Metrics

| Metric | Current | Minimum Target | Good Target |
|--------|---------|----------------|-------------|
| PSTH correlation | 0.03 | 0.30 | 0.50+ |
| Model mean FR | ~1 sp/s | 5-20 sp/s | within 20% of target |
| Model temporal std | ~0.2 | >0.5 | within 50% of target |
| L_neuron trend | flat | decreasing | decreasing then stable |

---

## Debugging Tips

### If PSTH correlation stays near zero:
1. Check that `rate_scale` is being learned (should increase from 10 toward ~15-20)
2. Verify inputs are being passed correctly (print shapes)
3. Try larger `W_in` initialization

### If loss explodes:
1. Reduce learning rate (try 1e-4)
2. Add gradient clipping (already present, but verify max_norm=1.0)
3. Check for NaN in model outputs

### If model outputs are constant across time:
1. Increase `spectral_radius` to 1.0 or 1.1 (more recurrent dynamics)
2. Verify noise is being injected (`noise_scale` > 0)
3. Check that inputs have temporal structure

### If train/val gap is large (overfitting):
1. Increase `lambda_reg` (try 1e-3)
2. Reduce model capacity (fewer hidden units)
3. Add dropout (not currently implemented, but could add)

---

## Implementation Order

1. **Run Priority 0** (diagnostic) to establish baseline
2. **Implement Priority 1** (architecture) - this is critical
3. **Implement Priority 2** (loss function) - this works with Priority 1
4. **Implement Priority 3** (data preprocessing) - quick win
5. **Run Test 2** to verify fixes are helping
6. **Implement Priority 4** (curriculum) if needed
7. **Run Test 4** (unconstrained) only if still failing

Do not skip Priority 1 and 2—they address the core failure mode.
