# Final Model Training Report

Generated: 2026-01-25 11:27:44

## Training Summary

| Metric | Value |
|--------|-------|
| Total epochs trained | 1259 |
| Best validation correlation | 0.4070 |
| Best epoch | 959 |
| Final validation correlation | 0.3943 |
| Early stopping | Yes |
| Total training time | 1:05:30 |

## Final Metrics

| Metric | Value |
|--------|-------|
| Overall PSTH correlation | 0.4003 |
| E neuron mean correlation | 0.3497 |
| I neuron mean correlation | 0.5599 |
| Model mean Fano factor | 3.16 |
| Real mean Fano factor | 49.49 |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| max_epochs | 2000 |
| patience | 300 |
| learning_rate | 0.001 |
| tau | 50.0 |
| noise_scale | 0.1 |
| input_embed_dim | 56 |
| input_embed_type | attention |
| attention_heads | 4 |
| learnable_h0 | True |
| h0_init | 0.1 |

## Data Configuration

| Parameter | Value |
|-----------|-------|
| Data file | data/rnn_export_Newton_08_15_2025_SC.mat |
| Total neurons | 54 (41 E, 13 I) |
| Train trials | 835 |
| Validation trials | 208 |
| Time bins | 219 |
| Bin size | 25.0 ms |

## Saved Files

### Model Checkpoints
- `model_best.pt` - Best validation correlation checkpoint
- `model_final.pt` - Final epoch checkpoint
- `checkpoints/model_epoch*.pt` - Periodic checkpoints

### Weight Matrices (weights/)
- `W_rec.npy` - Recurrent weights [65 x 65]
- `W_in.npy` - Input weights [65 x 56]
- `W_out.npy` - Output weights [2 x 52]
- `h0.npy` - Learned initial state [65]
- `attention_weights.npy` - Attention embedding weights
- `E_mask.npy` - Boolean mask for E neurons
- `I_mask.npy` - Boolean mask for I neurons

### Model Outputs (outputs/)
- `val_model_rates.npy` - Model firing rates [208 x 219 x 54]
- `val_target_rates.npy` - Target firing rates [208 x 219 x 54]
- `val_model_psth.npy` - Trial-averaged model PSTH [219 x 54]
- `val_target_psth.npy` - Trial-averaged target PSTH [219 x 54]
- `val_trial_conditions.npy` - Condition labels [208]

### Per-Neuron Metrics (metrics/)
- `per_neuron_correlation.npy` - PSTH correlation per neuron
- `per_neuron_mean_rate_model.npy` - Mean firing rate (model)
- `per_neuron_mean_rate_target.npy` - Mean firing rate (target)
- `per_neuron_variance_model.npy` - Temporal variance (model)
- `per_neuron_variance_target.npy` - Temporal variance (target)
- `per_neuron_fano_model.npy` - Fano factor (model)
- `per_neuron_fano_target.npy` - Fano factor (target)
- `neuron_ei_labels.npy` - E/I label (0=E, 1=I)

### Population Analysis (population/)
- `pca_real.npy` - PCA projections of real PSTH [219 x 10]
- `pca_model.npy` - PCA projections of model PSTH [219 x 10]
- `pca_components.npy` - PCA loading vectors [10 x 54]
- `pca_explained_variance.npy` - Variance explained per PC

### Training History
- `training_log.json` - Complete epoch-by-epoch training log
- `config.json` - Full configuration used

### Visualizations (figures/)
- `training_curves.png` - Loss and correlation over training
- `per_neuron_correlation_histogram.png` - Correlation distribution by E/I
- `per_neuron_correlation_scatter.png` - Correlation vs firing rate
- `best_worst_neurons.png` - PSTHs for best/worst fitting neurons
- `weight_matrices.png` - Heatmaps of W_rec and W_in
- `h0_distribution.png` - Learned initial state distribution
- `attention_patterns.png` - Attention embedding weights
- `pca_trajectories.png` - PC1-3 over time
- `pca_state_space.png` - PC1 vs PC2 phase plot
- `fano_factor_comparison.png` - Model vs real Fano factors

## Code Snippets for Loading Data

```python
import numpy as np
import torch

# Load weights
W_rec = np.load('results/final_model/weights/W_rec.npy')
W_in = np.load('results/final_model/weights/W_in.npy')
W_out = np.load('results/final_model/weights/W_out.npy')
h0 = np.load('results/final_model/weights/h0.npy')
E_mask = np.load('results/final_model/weights/E_mask.npy')
I_mask = np.load('results/final_model/weights/I_mask.npy')

# Load model outputs
model_rates = np.load('results/final_model/outputs/val_model_rates.npy')
target_rates = np.load('results/final_model/outputs/val_target_rates.npy')
model_psth = np.load('results/final_model/outputs/val_model_psth.npy')
target_psth = np.load('results/final_model/outputs/val_target_psth.npy')

# Load per-neuron metrics
correlations = np.load('results/final_model/metrics/per_neuron_correlation.npy')
ei_labels = np.load('results/final_model/metrics/neuron_ei_labels.npy')

# Load PCA results
pca_real = np.load('results/final_model/population/pca_real.npy')
pca_model = np.load('results/final_model/population/pca_model.npy')
pca_explained = np.load('results/final_model/population/pca_explained_variance.npy')

# Load trained model
from src.model import create_model_from_data
model = create_model_from_data(
    n_classic=41,
    n_interneuron=13,
    n_inputs=14,
    enforce_ratio=True,
    input_embed_dim=56,
    input_embed_type='attention',
    learnable_h0=True,
    device='cpu'
)
model.load_state_dict(torch.load('results/final_model/model_best.pt'))
model.eval()
```
