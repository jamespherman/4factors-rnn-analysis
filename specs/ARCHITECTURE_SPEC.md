# ARCHITECTURE_SPEC.md - E-I RNN Model Architecture

## Overview

This document specifies the excitatory-inhibitory recurrent neural network architecture for fitting SC neural activity. The model follows the framework of Song et al. (2016) with constraints validated by Sourmpis et al. (2026).

## Model Equations

### State Dynamics

The network state evolves according to:

```
τ · dx/dt = -x + W_rec · r + W_in · u + σξ
```

In discrete time (Euler integration with dt = bin_size):

```
x[t] = (1 - dt/τ) · x[t-1] + (dt/τ) · (W_rec · r[t-1] + W_in · u[t]) + σ · √(dt/τ) · ξ[t]
```

Where:
- `x[t]` ∈ ℝ^N: Neural state (membrane potential / activation)
- `r[t]` ∈ ℝ^N: Firing rate, `r = φ(x)`
- `u[t]` ∈ ℝ^M: Input signals
- `W_rec` ∈ ℝ^(N×N): Recurrent weight matrix
- `W_in` ∈ ℝ^(N×M): Input weight matrix
- `τ`: Time constant (50ms, following Song et al.)
- `σ`: Noise scale
- `ξ[t]`: Standard normal noise

### Activation Function

```
r = φ(x) = softplus(x) = log(1 + exp(x))
```

**Rationale**: Softplus is smooth (unlike ReLU), always positive (firing rates can't be negative), and approximates ReLU for large x. This matches RajaKumar et al. (2021).

### Output (Behavioral Prediction)

```
y[t] = W_out · r[t] + b_out
```

Where:
- `y[t]` ∈ ℝ^2: Predicted eye position (x, y)
- `W_out` ∈ ℝ^(2×N_exc): Readout from excitatory neurons only
- `b_out` ∈ ℝ^2: Output bias

**Note**: Only excitatory neurons contribute to motor output, following biological constraint that SC output neurons are excitatory.

## Network Structure

### Unit Counts

For a session with `N_classic` classic SC neurons and `N_intern` putative interneurons:

```
N_exc_recorded = N_classic           # Recorded excitatory
N_inh_recorded = N_intern            # Recorded inhibitory

# Add hidden units to maintain 4:1 ratio
N_total = max(N_classic + N_intern, ceil((N_classic + N_intern) / 0.8))
N_exc_total = ceil(0.8 * N_total)
N_inh_total = N_total - N_exc_total

N_exc_hidden = N_exc_total - N_exc_recorded
N_inh_hidden = N_inh_total - N_inh_recorded
```

### Unit Ordering Convention

```
Units 0 to N_exc_recorded-1:          Recorded excitatory (classic SC)
Units N_exc_recorded to N_exc_total-1: Hidden excitatory
Units N_exc_total to N_exc_total+N_inh_recorded-1: Recorded inhibitory (putative INT)
Units N_exc_total+N_inh_recorded to N_total-1:     Hidden inhibitory
```

### Sign Mask for Dale's Law

```python
# Create sign mask: +1 for E columns, -1 for I columns
sign_mask = torch.ones(N_total, N_total)
sign_mask[:, N_exc_total:] = -1  # Inhibitory neurons have negative outgoing weights

# Enforce Dale's law
W_rec = torch.abs(W_raw) * sign_mask
```

### Local Inhibition Constraint

Inhibitory neurons project only within SC (no output connections):

```python
# Mask for W_out: only excitatory neurons contribute
output_mask = torch.zeros(2, N_total)
output_mask[:, :N_exc_total] = 1

W_out = W_out_raw * output_mask
```

For multi-area models (future extension), inhibitory neurons would not project across areas. For single-area SC, this constraint means I→output weights are zero.

## Input Structure

### Input Dimensions

```
M = 14  # Total input dimension

Inputs (all binary except eye position):
- input_fixation_on:    1 dim
- input_target_loc:     4 dims (one-hot)
- input_go_signal:      1 dim
- input_reward_on:      1 dim
- input_eye_x:          1 dim (continuous, normalized)
- input_eye_y:          1 dim (continuous, normalized)
- input_is_face:        1 dim
- input_is_nonface:     1 dim
- input_is_bullseye:    1 dim
- input_high_salience:  1 dim
- input_low_salience:   1 dim
```

### Input Normalization

```python
# Binary inputs: use as-is (0 or 1)
# Eye position: z-score normalize across session
eye_x_norm = (eye_x - eye_x.mean()) / eye_x.std()
eye_y_norm = (eye_y - eye_y.mean()) / eye_y.std()
```

## Weight Initialization

Following Sourmpis et al. (2026) and Rajan & Abbott (2006):

### Recurrent Weights

```python
def initialize_W_rec(N_exc, N_inh, spectral_radius=0.9):
    """Initialize recurrent weights with balanced E/I and controlled spectral radius."""
    N = N_exc + N_inh
    
    # Initialize from gamma distribution (positive values)
    W_raw = np.random.gamma(shape=2, scale=0.05, size=(N, N))
    
    # Zero diagonal (no self-connections)
    np.fill_diagonal(W_raw, 0)
    
    # Balance E/I: scale inhibitory weights so sum(E) ≈ sum(I) for each row
    E_sum = W_raw[:, :N_exc].sum(axis=1, keepdims=True)
    I_sum = W_raw[:, N_exc:].sum(axis=1, keepdims=True)
    W_raw[:, N_exc:] *= (E_sum / (I_sum + 1e-8))
    
    # Apply sign mask
    sign_mask = np.ones((N, N))
    sign_mask[:, N_exc:] = -1
    W_rec = W_raw * sign_mask
    
    # Scale to target spectral radius
    eigenvalues = np.linalg.eigvals(W_rec)
    current_radius = np.max(np.abs(eigenvalues))
    W_rec *= (spectral_radius / current_radius)
    
    return torch.tensor(W_rec, dtype=torch.float32)
```

### Input Weights

```python
def initialize_W_in(N, M):
    """Initialize input weights from uniform distribution."""
    W_in = np.random.uniform(-0.1, 0.1, size=(N, M))
    return torch.tensor(W_in, dtype=torch.float32)
```

### Output Weights

```python
def initialize_W_out(N_exc):
    """Initialize output weights (only from excitatory neurons)."""
    W_out = np.random.uniform(-0.1, 0.1, size=(2, N_exc))
    return torch.tensor(W_out, dtype=torch.float32)
```

## Hyperparameters

| Parameter | Value | Source/Rationale |
|-----------|-------|------------------|
| τ (time constant) | 50 ms | Song et al. (2016), RajaKumar et al. (2021) |
| dt (integration step) | 25 ms | Matches data bin size |
| σ (noise scale) | 0.01 | Sourmpis et al. (2026) |
| Spectral radius | 0.9 | Standard for stable RNNs |
| E:I ratio | 4:1 (80:20) | Dale's principle, Sourmpis et al. |

## PyTorch Implementation Skeleton

```python
import torch
import torch.nn as nn

class EIRNN(nn.Module):
    """Excitatory-Inhibitory RNN with Dale's law constraints."""
    
    def __init__(self, n_exc, n_inh, n_inputs, tau=50.0, dt=25.0, noise_scale=0.01):
        super().__init__()
        
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.n_inputs = n_inputs
        self.tau = tau
        self.dt = dt
        self.noise_scale = noise_scale
        self.alpha = dt / tau  # Integration constant
        
        # Learnable parameters (raw, unconstrained)
        self.W_rec_raw = nn.Parameter(torch.zeros(self.n_total, self.n_total))
        self.W_in = nn.Parameter(torch.zeros(self.n_total, n_inputs))
        self.W_out = nn.Parameter(torch.zeros(2, n_exc))  # Only E neurons
        self.b_out = nn.Parameter(torch.zeros(2))
        
        # Fixed sign mask for Dale's law
        sign_mask = torch.ones(self.n_total, self.n_total)
        sign_mask[:, n_exc:] = -1
        self.register_buffer('sign_mask', sign_mask)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # ... initialization code from above ...
        pass
    
    @property
    def W_rec(self):
        """Recurrent weights with Dale's law enforced."""
        W = torch.abs(self.W_rec_raw) * self.sign_mask
        # Zero diagonal
        W = W * (1 - torch.eye(self.n_total, device=W.device))
        return W
    
    def forward(self, inputs, initial_state=None):
        """
        Args:
            inputs: [batch, time, n_inputs]
            initial_state: [batch, n_total] or None
        
        Returns:
            rates: [batch, time, n_total] - Firing rates
            outputs: [batch, time, 2] - Eye position predictions
        """
        batch_size, n_steps, _ = inputs.shape
        device = inputs.device
        
        # Initialize state
        if initial_state is None:
            x = torch.zeros(batch_size, self.n_total, device=device)
        else:
            x = initial_state
        
        # Get constrained weights
        W_rec = self.W_rec
        
        # Storage
        rates_list = []
        outputs_list = []
        
        for t in range(n_steps):
            # Current firing rate
            r = torch.nn.functional.softplus(x)
            
            # Recurrent input
            rec_input = torch.matmul(r, W_rec.T)
            
            # External input
            ext_input = torch.matmul(inputs[:, t, :], self.W_in.T)
            
            # Noise
            noise = self.noise_scale * torch.randn_like(x) * (self.alpha ** 0.5)
            
            # State update (Euler integration)
            x = (1 - self.alpha) * x + self.alpha * (rec_input + ext_input) + noise
            
            # Output (from excitatory neurons only)
            r_exc = r[:, :self.n_exc]
            y = torch.matmul(r_exc, self.W_out.T) + self.b_out
            
            rates_list.append(r)
            outputs_list.append(y)
        
        rates = torch.stack(rates_list, dim=1)
        outputs = torch.stack(outputs_list, dim=1)
        
        return rates, outputs
```

## Model Variants for Comparison

### Variant A: Full Constraints (Primary)
- Dale's law: ✓
- Local inhibition: ✓
- 4:1 ratio: ✓
- Recorded neurons assigned by classification

### Variant B: Unconstrained E/I
- Dale's law: ✗ (learn signs)
- Local inhibition: ✗
- Used to test if functional classification matches physiological

### Variant C: Classic Neurons Only
- Include only classic SC neurons
- Network must "invent" inhibitory dynamics
- Compare invented I units to actual putative interneurons

## Computational Considerations

### Memory
- For N=100 neurons, T=150 bins, B=200 trials:
- State storage: N × T × B × 4 bytes ≈ 12 MB per forward pass
- Weight storage: N² × 4 bytes ≈ 40 KB
- Feasible on any modern GPU

### Speed
- Single forward pass: ~10ms on GPU
- Full training (1000 epochs, 200 trials): ~30 minutes on GPU

## References

- Song, H.F., Yang, G.R., & Wang, X.J. (2016). Training excitatory-inhibitory recurrent neural networks for cognitive tasks. PLoS Computational Biology.
- RajaKumar, A., Rinzel, J., & Chen, Z.S. (2021). Stimulus-driven and spontaneous dynamics in excitatory-inhibitory recurrent neural networks. Neural Computation.
- Sourmpis, C., Petersen, C.C., Gerstner, W., & Bellec, G. (2026). Biologically informed cortical models predict optogenetic perturbations. eLife.
