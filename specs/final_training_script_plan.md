# Final Training Script Plan

## Overview

This document outlines the plan for `scripts/train_final_model.py`, a comprehensive self-contained training script that produces all data needed for scientific analysis of the learned E-I RNN weights.

## What the Script Will Do

1. **Load data** from `data/rnn_export_Newton_08_15_2025_SC.mat`
2. **Configure model** using best Phase 5 config (attention embedding + learnable h0)
3. **Train** for up to 2000 epochs with comprehensive logging
4. **Save** all weights, outputs, and metrics for downstream analysis
5. **Generate** visualizations summarizing training and model performance
6. **Produce** a final markdown report with all results

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| max_epochs | 2000 | Thorough training |
| patience | 300 | Longer patience for convergence |
| learning_rate | 1e-3 | Adam optimizer |
| lr_scheduler | ReduceLROnPlateau | factor=0.5, patience=50, min_lr=1e-5 |
| gradient_clip | 1.0 | Prevent exploding gradients |

### Model Configuration (Best Phase 5)

| Parameter | Value |
|-----------|-------|
| input_embed_dim | 56 |
| input_embed_type | 'attention' |
| attention_heads | 4 |
| learnable_h0 | True |
| h0_init | 0.1 |
| tau | 50.0 (fixed) |
| noise_scale | 0.1 |
| dt | 25.0 ms |
| spectral_radius | 0.9 |

### Loss Configuration

| Parameter | Value |
|-----------|-------|
| use_grad_balancing | True |
| ltrial_scale | 0.5 |
| lambda_reg | 1e-4 |
| lambda_scale | 0.1 |
| lambda_var | 0.05 |

## Data Logged During Training

### Every Epoch (to `training_log.json`)

- `epoch`: Current epoch number
- `train_loss`: Total training loss
- `val_loss`: Validation L_neuron loss
- `val_correlation`: Mean PSTH correlation on validation set
- `learning_rate`: Current learning rate
- `epoch_time`: Time in seconds for this epoch
- `L_neuron`: Neuron-wise PSTH loss
- `L_trial`: Trial-matching loss
- `L_reg`: Regularization loss

### Every 100 Epochs

- `per_neuron_correlations`: Correlation for each neuron [n_neurons]
- `E_mean_corr`: Mean correlation for E neurons
- `I_mean_corr`: Mean correlation for I neurons
- `h0_stats`: Mean, std, min, max of learned h0
- `fano_model`: Model Fano factor
- `fano_real`: Target Fano factor

## Files Saved at End of Training

All files saved to `results/final_model/`:

### Model Checkpoints

| File | Description |
|------|-------------|
| `model_best.pt` | Best validation correlation checkpoint |
| `model_final.pt` | Final epoch checkpoint |
| `checkpoints/model_epoch{N}.pt` | Checkpoint every 100 epochs |

### Weight Matrices (`weights/`)

| File | Shape | Description |
|------|-------|-------------|
| `W_rec.npy` | [n_neurons, n_neurons] | Recurrent weights (Dale's law applied) |
| `W_in.npy` | [n_neurons, embed_dim] | Input weights |
| `W_out.npy` | [n_outputs, n_exc] | Output weights |
| `h0.npy` | [n_neurons] | Learned initial state |
| `attention_weights.npy` | dict | Q, K, V projection weights |
| `input_embed_linear.npy` | [embed_dim, n_inputs*heads] | Attention output projection |
| `E_mask.npy` | [n_neurons] | Boolean mask for E neurons |
| `I_mask.npy` | [n_neurons] | Boolean mask for I neurons |

### Model Outputs (`outputs/`)

| File | Shape | Description |
|------|-------|-------------|
| `val_model_rates.npy` | [n_val_trials, n_time, n_neurons] | Model firing rates on validation |
| `val_target_rates.npy` | [n_val_trials, n_time, n_neurons] | Target firing rates on validation |
| `val_model_psth.npy` | [n_time, n_neurons] | Trial-averaged model PSTH |
| `val_target_psth.npy` | [n_time, n_neurons] | Trial-averaged target PSTH |
| `val_trial_conditions.npy` | [n_val_trials] | Condition labels per trial |

### Per-Neuron Metrics (`metrics/`)

| File | Shape | Description |
|------|-------|-------------|
| `per_neuron_correlation.npy` | [n_neurons] | PSTH correlation per neuron |
| `per_neuron_mean_rate_model.npy` | [n_neurons] | Mean firing rate (model) |
| `per_neuron_mean_rate_target.npy` | [n_neurons] | Mean firing rate (target) |
| `per_neuron_variance_model.npy` | [n_neurons] | Temporal variance (model) |
| `per_neuron_variance_target.npy` | [n_neurons] | Temporal variance (target) |
| `per_neuron_fano_model.npy` | [n_neurons] | Fano factor (model) |
| `per_neuron_fano_target.npy` | [n_neurons] | Fano factor (target) |
| `neuron_ei_labels.npy` | [n_neurons] | E/I label (0=E, 1=I) |

### Population Analysis (`population/`)

| File | Shape | Description |
|------|-------|-------------|
| `pca_real.npy` | [n_time, n_pcs] | PCA projections of real PSTH |
| `pca_model.npy` | [n_time, n_pcs] | PCA projections of model PSTH |
| `pca_components.npy` | [n_pcs, n_neurons] | PCA loading vectors |
| `pca_explained_variance.npy` | [n_pcs] | Variance explained per PC |

### Training History

| File | Description |
|------|-------------|
| `training_log.json` | Complete epoch-by-epoch training log |
| `config.json` | Full configuration used |

## Visualizations Generated (`figures/`)

### Training Curves (`training_curves.png`)
- 4-panel figure:
  - Train/val loss vs epoch
  - Val correlation vs epoch
  - Learning rate vs epoch
  - Best correlation marker

### Per-Neuron Analysis
- `per_neuron_correlation_histogram.png`: Histogram of correlations, colored by E/I
- `per_neuron_correlation_scatter.png`: Correlation vs mean firing rate, colored by E/I

### PSTH Examples (`best_worst_neurons.png`)
- 2x4 grid showing PSTHs for 4 best and 4 worst fitting neurons
- Model vs target comparison

### Weight Visualization (`weight_matrices.png`)
- Heatmaps of W_rec and W_in sorted by E/I

### Initial State (`h0_distribution.png`)
- Histogram of learned h0 values, colored by E/I

### Attention Patterns (`attention_patterns.png`)
- Visualization of learned attention weights (Q, K, V heatmaps)

### Population Analysis
- `pca_trajectories.png`: PC1-3 over time, real vs model
- `pca_state_space.png`: PC1 vs PC2 phase plot, real vs model trajectories

### Fano Factor (`fano_factor_comparison.png`)
- Model vs real Fano factors with log scale axes
- Show both distributions side-by-side

## Final Report Structure (`training_report.md`)

```markdown
# Final Model Training Report

## Training Summary
- Total epochs trained
- Best validation correlation (and epoch)
- Final validation correlation
- Early stopping info (if applicable)
- Total training time

## Final Metrics
- Overall PSTH correlation: X.XXX
- E neuron mean correlation: X.XXX
- I neuron mean correlation: X.XXX
- Model mean Fano factor: X.XX
- Real mean Fano factor: X.XX

## Model Configuration
[Table of all configuration parameters]

## Saved Files
[List of all files with descriptions]

## Code Snippets for Analysis
[Ready-to-use Python code for loading data]
```

## Console Output During Training

### Every Epoch (progress bar)
```
Epoch 100/2000: 100%|████| [12:34] train_loss=0.234 val_corr=0.412 lr=1.0e-03
```

### Every 10 Epochs
```
Epoch 100 | train_loss: 0.234 | val_corr: 0.412 | lr: 1.0e-03 | time: 5.2s
```

### Every 100 Epochs
```
=== Epoch 100 Detailed Report ===
  E neurons: mean_corr=0.452 (n=42)
  I neurons: mean_corr=0.301 (n=8)
  Fano factor: model=2.34, real=3.12
  h0: mean=0.51, std=0.47
  Saved checkpoint: checkpoints/model_epoch100.pt
```

### On Completion
```
================================================================================
TRAINING COMPLETE
================================================================================
Total epochs: 850 (early stopped at patience=300)
Best validation correlation: 0.4121 (epoch 550)
Final validation correlation: 0.4087
Training time: 2h 34m 12s

Files saved to: results/final_model/
See training_report.md for full summary.
================================================================================
```

## Usage

```bash
# Run with all defaults (2000 epochs, patience=300)
python scripts/train_final_model.py

# Quick test run (5 epochs)
python scripts/train_final_model.py --test
```

## Implementation Notes

1. **No command-line arguments for config**: All training parameters are hardcoded for reproducibility of this final run.

2. **Deterministic seeding**: Uses seed=42 for reproducibility.

3. **GPU support**: Automatically uses CUDA if available.

4. **Checkpoint saving**: Every 100 epochs + best + final.

5. **Graceful interruption**: Saves current state if interrupted (Ctrl+C).

6. **Self-contained**: All imports are from the existing codebase (`src/`).
