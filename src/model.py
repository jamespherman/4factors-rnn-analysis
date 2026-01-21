"""
Excitatory-Inhibitory RNN with Dale's Law constraints.

Based on Song et al. (2016) and Sourmpis et al. (2026).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class EIRNN(nn.Module):
    """
    Excitatory-Inhibitory RNN with Dale's law constraints.
    
    Architecture:
        - N = n_exc + n_inh total units
        - Dale's law: E units have positive outgoing weights, I units have negative
        - Local inhibition: I units don't project to output (only E units do)
        - Rate-based dynamics with softplus activation
    
    Dynamics:
        x[t] = (1 - α) * x[t-1] + α * (W_rec @ r[t-1] + W_in @ u[t]) + noise
        r[t] = softplus(x[t])
        y[t] = W_out @ r_exc[t] + b_out
    
    where α = dt/τ
    """
    
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
        device: str = 'cpu'
    ):
        """
        Initialize E-I RNN.
        
        Args:
            n_exc: Number of excitatory units
            n_inh: Number of inhibitory units
            n_inputs: Dimension of input signals
            n_outputs: Dimension of output (default 2 for eye x,y)
            tau: Time constant in ms
            dt: Integration timestep in ms (should match data bin size)
            noise_scale: Standard deviation of noise
            spectral_radius: Target spectral radius for weight initialization
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau
        self.noise_scale = noise_scale
        self.spectral_radius = spectral_radius
        
        # Learnable parameters (raw, unconstrained)
        self.W_rec_raw = nn.Parameter(torch.zeros(self.n_total, self.n_total))
        self.W_in = nn.Parameter(torch.zeros(self.n_total, n_inputs))
        self.W_out = nn.Parameter(torch.zeros(n_outputs, n_exc))  # Only E neurons
        self.b_out = nn.Parameter(torch.zeros(n_outputs))
        
        # Fixed sign mask for Dale's law: +1 for E columns, -1 for I columns
        sign_mask = torch.ones(self.n_total, self.n_total)
        sign_mask[:, n_exc:] = -1
        self.register_buffer('sign_mask', sign_mask)
        
        # Diagonal mask (no self-connections)
        diag_mask = 1 - torch.eye(self.n_total)
        self.register_buffer('diag_mask', diag_mask)
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(device)
    
    def _initialize_weights(self):
        """Initialize weights with balanced E/I and controlled spectral radius."""
        n_exc = self.n_exc
        n_inh = self.n_inh
        n_total = self.n_total
        
        # Initialize from gamma distribution (positive values)
        W_raw = np.random.gamma(shape=2, scale=0.05, size=(n_total, n_total))
        
        # Zero diagonal
        np.fill_diagonal(W_raw, 0)
        
        # Balance E/I: scale so sum(E inputs) ≈ sum(I inputs) for each neuron
        if n_inh > 0:
            E_sum = W_raw[:, :n_exc].sum(axis=1, keepdims=True)
            I_sum = W_raw[:, n_exc:].sum(axis=1, keepdims=True)
            scale_factor = E_sum / (I_sum + 1e-8)
            W_raw[:, n_exc:] *= scale_factor
        
        # Apply sign mask
        sign_mask_np = self.sign_mask.cpu().numpy()
        W_rec = W_raw * sign_mask_np
        
        # Scale to target spectral radius
        eigenvalues = np.linalg.eigvals(W_rec)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_rec *= (self.spectral_radius / current_radius)
        
        # Store as raw (absolute values, sign will be applied in forward)
        self.W_rec_raw.data = torch.tensor(np.abs(W_rec), dtype=torch.float32)
        
        # Initialize input weights
        nn.init.uniform_(self.W_in, -0.1, 0.1)
        
        # Initialize output weights
        nn.init.uniform_(self.W_out, -0.1, 0.1)
    
    @property
    def W_rec(self) -> torch.Tensor:
        """Recurrent weights with Dale's law enforced."""
        W = torch.abs(self.W_rec_raw) * self.sign_mask * self.diag_mask
        return W
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        initial_state: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            inputs: [batch, time, n_inputs] - Input signals
            initial_state: [batch, n_total] - Initial membrane potentials (optional)
            return_states: Whether to return membrane potentials (for analysis)
        
        Returns:
            rates: [batch, time, n_total] - Firing rates
            outputs: [batch, time, n_outputs] - Output predictions (e.g., eye position)
            states: [batch, time, n_total] - Membrane potentials (if return_states=True)
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
        states_list = [] if return_states else None
        
        for t in range(n_steps):
            # Current firing rate
            r = torch.nn.functional.softplus(x)
            
            # Recurrent input: r @ W_rec.T gives [batch, n_total]
            rec_input = torch.matmul(r, W_rec.T)
            
            # External input
            ext_input = torch.matmul(inputs[:, t, :], self.W_in.T)
            
            # Noise (scaled by sqrt(alpha) for proper discrete-time variance)
            noise = self.noise_scale * torch.randn_like(x) * (self.alpha ** 0.5)
            
            # State update (Euler integration)
            x = (1 - self.alpha) * x + self.alpha * (rec_input + ext_input) + noise
            
            # Output (from excitatory neurons only)
            r_exc = r[:, :self.n_exc]
            y = torch.matmul(r_exc, self.W_out.T) + self.b_out
            
            rates_list.append(r)
            outputs_list.append(y)
            if return_states:
                states_list.append(x.clone())
        
        rates = torch.stack(rates_list, dim=1)
        outputs = torch.stack(outputs_list, dim=1)
        
        if return_states:
            states = torch.stack(states_list, dim=1)
            return rates, outputs, states
        
        return rates, outputs
    
    def get_weight_submatrices(self) -> dict:
        """
        Extract weight submatrices for analysis.
        
        Returns:
            dict with keys:
                - W_EE: E→E weights [n_exc, n_exc]
                - W_EI: I→E weights [n_exc, n_inh] (these are the key ones!)
                - W_IE: E→I weights [n_inh, n_exc]
                - W_II: I→I weights [n_inh, n_inh]
        """
        W = self.W_rec.detach().cpu().numpy()
        n_exc = self.n_exc
        
        return {
            'W_EE': W[:n_exc, :n_exc],           # E→E (positive)
            'W_EI': W[:n_exc, n_exc:],           # I→E (negative) - KEY FOR ANALYSIS
            'W_IE': W[n_exc:, :n_exc],           # E→I (positive)
            'W_II': W[n_exc:, n_exc:],           # I→I (negative)
        }
    
    def verify_constraints(self) -> bool:
        """Verify Dale's law and other constraints are satisfied."""
        W = self.W_rec.detach().cpu().numpy()
        n_exc = self.n_exc
        
        # Check E columns are non-negative
        e_cols_ok = (W[:, :n_exc] >= -1e-6).all()
        
        # Check I columns are non-positive  
        i_cols_ok = (W[:, n_exc:] <= 1e-6).all()
        
        # Check diagonal is zero
        diag_ok = np.allclose(np.diag(W), 0, atol=1e-6)
        
        if not e_cols_ok:
            print("WARNING: E columns have negative values")
        if not i_cols_ok:
            print("WARNING: I columns have positive values")
        if not diag_ok:
            print("WARNING: Diagonal is not zero")
        
        return e_cols_ok and i_cols_ok and diag_ok


def create_model_from_data(
    n_classic: int,
    n_interneuron: int,
    n_inputs: int,
    enforce_ratio: bool = True,
    target_ratio: float = 4.0,
    **kwargs
) -> EIRNN:
    """
    Create model with appropriate E/I counts based on recorded neurons.
    
    Args:
        n_classic: Number of classic SC neurons (assigned to E)
        n_interneuron: Number of putative interneurons (assigned to I)
        n_inputs: Number of input dimensions
        enforce_ratio: Whether to add hidden units for 4:1 ratio
        target_ratio: Target E:I ratio (default 4:1)
        **kwargs: Additional arguments passed to EIRNN
    
    Returns:
        EIRNN model
    """
    n_recorded = n_classic + n_interneuron
    
    if enforce_ratio:
        # Calculate total units needed to achieve target ratio
        # with at least n_classic E and n_interneuron I
        
        # If we have too many I relative to E, add hidden E
        # If we have too many E relative to I, add hidden I
        
        current_ratio = n_classic / max(n_interneuron, 1)
        
        if current_ratio >= target_ratio:
            # Need more I units
            n_exc = n_classic
            n_inh = int(np.ceil(n_classic / target_ratio))
            n_inh = max(n_inh, n_interneuron)  # At least as many as recorded
        else:
            # Need more E units
            n_inh = n_interneuron
            n_exc = int(np.ceil(n_interneuron * target_ratio))
            n_exc = max(n_exc, n_classic)  # At least as many as recorded
        
        n_hidden_exc = n_exc - n_classic
        n_hidden_inh = n_inh - n_interneuron
        
        print(f"Model configuration:")
        print(f"  Recorded E (classic): {n_classic}")
        print(f"  Recorded I (interneuron): {n_interneuron}")
        print(f"  Hidden E: {n_hidden_exc}")
        print(f"  Hidden I: {n_hidden_inh}")
        print(f"  Total: {n_exc} E + {n_inh} I = {n_exc + n_inh}")
        print(f"  E:I ratio: {n_exc/n_inh:.2f}")
    else:
        n_exc = n_classic
        n_inh = n_interneuron
    
    return EIRNN(n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing EIRNN...")
    
    model = EIRNN(n_exc=80, n_inh=20, n_inputs=14, device='cpu')
    
    # Verify constraints
    assert model.verify_constraints(), "Constraints violated!"
    
    # Test forward pass
    batch_size = 32
    n_steps = 100
    inputs = torch.randn(batch_size, n_steps, 14)
    
    rates, outputs = model(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Rates shape: {rates.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Check weight submatrices
    weights = model.get_weight_submatrices()
    print(f"\nWeight submatrices:")
    for name, W in weights.items():
        print(f"  {name}: {W.shape}, range [{W.min():.3f}, {W.max():.3f}]")
    
    print("\nAll tests passed!")
