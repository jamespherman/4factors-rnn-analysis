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
        noise_scale: float = 0.1,
        spectral_radius: float = 0.9,
        bypass_dale: bool = False,
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
            noise_scale: Standard deviation of noise (default 0.1 for better PSTH fitting)
            spectral_radius: Target spectral radius for weight initialization
            bypass_dale: If True, disable Dale's law constraints (for debugging)
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
        self.bypass_dale = bypass_dale

        # Learnable parameters (raw, unconstrained)
        self.W_rec_raw = nn.Parameter(torch.zeros(self.n_total, self.n_total))
        self.W_in = nn.Parameter(torch.zeros(self.n_total, n_inputs))
        self.W_out = nn.Parameter(torch.zeros(n_outputs, n_exc))  # Only E neurons
        self.b_out = nn.Parameter(torch.zeros(n_outputs))

        # Learnable rate scaling (initialize near target mean firing rate)
        # This addresses the scale mismatch: softplus outputs ~1 sp/s, target is ~11 sp/s
        self.rate_scale = nn.Parameter(torch.tensor(10.0))
        self.rate_baseline = nn.Parameter(torch.tensor(0.5))
        
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

        if self.bypass_dale:
            # For unconstrained network: use standard random initialization
            # with zero mean and controlled spectral radius
            W_rec = np.random.randn(n_total, n_total) / np.sqrt(n_total)
            np.fill_diagonal(W_rec, 0)

            # Scale to target spectral radius
            eigenvalues = np.linalg.eigvals(W_rec)
            current_radius = np.max(np.abs(eigenvalues))
            if current_radius > 0:
                W_rec *= (self.spectral_radius / current_radius)

            # Store directly (no absolute value needed)
            self.W_rec_raw.data = torch.tensor(W_rec, dtype=torch.float32)
        else:
            # For E-I constrained network: initialize with gamma distribution
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

        # Initialize input weights - moderate positive bias for excitatory drive
        nn.init.uniform_(self.W_in, 0.0, 1.0)

        # Initialize output weights
        nn.init.uniform_(self.W_out, -0.1, 0.1)
    
    @property
    def W_rec(self) -> torch.Tensor:
        """Recurrent weights with Dale's law enforced (unless bypassed)."""
        if self.bypass_dale:
            # No sign constraints - weights can be positive or negative
            W = self.W_rec_raw * self.diag_mask
        else:
            # Apply Dale's law: |W| * sign_mask * diag_mask
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
            # Raw firing rate (internal dynamics use unscaled rates)
            r_raw = torch.nn.functional.softplus(x)

            # Recurrent input uses RAW rates to maintain stable dynamics
            # The rate_scale only affects the OUTPUT, not the internal feedback
            rec_input = torch.matmul(r_raw, W_rec.T)

            # External input
            ext_input = torch.matmul(inputs[:, t, :], self.W_in.T)

            # Noise (scaled by sqrt(alpha) for proper discrete-time variance)
            noise = self.noise_scale * torch.randn_like(x) * (self.alpha ** 0.5)

            # State update (Euler integration)
            x = (1 - self.alpha) * x + self.alpha * (rec_input + ext_input) + noise

            # Apply learnable scale and baseline for OUTPUT rates only
            # This matches the target data scale without destabilizing dynamics
            r_scaled = r_raw * self.rate_scale + self.rate_baseline
            r_scaled = torch.clamp(r_scaled, min=0.0)  # Ensure non-negative

            # Output (from excitatory neurons only, using scaled rates)
            r_exc = r_scaled[:, :self.n_exc]
            y = torch.matmul(r_exc, self.W_out.T) + self.b_out

            rates_list.append(r_scaled)
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
    bypass_dale: bool = False,
    target_total: int = None,
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
        bypass_dale: If True, disable Dale's law constraints
        target_total: If specified, target this many total units (overrides enforce_ratio logic)
                     Uses target_ratio to determine E/I split (e.g., 200 total with 4:1 = 160E + 40I)
        **kwargs: Additional arguments passed to EIRNN

    Returns:
        EIRNN model
    """
    n_recorded = n_classic + n_interneuron

    if target_total is not None:
        # Use target_total with target_ratio to determine E/I split
        # E/(E+I) = ratio/(ratio+1), I/(E+I) = 1/(ratio+1)
        n_exc = int(round(target_total * target_ratio / (target_ratio + 1)))
        n_inh = target_total - n_exc

        # Ensure we have at least as many as recorded
        if n_exc < n_classic:
            print(f"WARNING: target_total {target_total} gives only {n_exc} E units, but {n_classic} recorded. Adjusting.")
            n_exc = n_classic
            n_inh = max(n_inh, n_interneuron)
        if n_inh < n_interneuron:
            print(f"WARNING: target_total {target_total} gives only {n_inh} I units, but {n_interneuron} recorded. Adjusting.")
            n_inh = n_interneuron

        n_hidden_exc = n_exc - n_classic
        n_hidden_inh = n_inh - n_interneuron

        print(f"Model configuration (target_total={target_total}):")
        print(f"  Recorded E (classic): {n_classic}")
        print(f"  Recorded I (interneuron): {n_interneuron}")
        print(f"  Hidden E: {n_hidden_exc}")
        print(f"  Hidden I: {n_hidden_inh}")
        print(f"  Total: {n_exc} E + {n_inh} I = {n_exc + n_inh}")
        print(f"  E:I ratio: {n_exc/n_inh:.2f}")
        if bypass_dale:
            print(f"  *** BYPASS_DALE=True: Dale's law constraints DISABLED ***")
    elif enforce_ratio:
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
        if bypass_dale:
            print(f"  *** BYPASS_DALE=True: Dale's law constraints DISABLED ***")
    else:
        n_exc = n_classic
        n_inh = n_interneuron

    return EIRNN(n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, bypass_dale=bypass_dale, **kwargs)


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
