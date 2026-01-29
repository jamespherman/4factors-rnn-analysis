"""
Extended model architectures for E-I RNN experiments.

Import these alongside the original model:
    from src.model import EIRNN, create_model_from_data
    from src.model_extended import EIRNN_LearnableTau, EIRNN_LowRank, create_model_from_data_extended
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from src.model import EIRNN


class EIRNN_LearnableTau(EIRNN):
    """
    EIRNN with learnable time constants.
    
    Options:
    - tau_mode='per_pop': Separate tau for E and I populations (2 params)
    - tau_mode='per_neuron': Per-neuron tau (n_total params)
    """
    
    def __init__(
        self,
        n_exc: int,
        n_inh: int,
        n_inputs: int,
        n_outputs: int = 2,
        tau_e_init: float = 50.0,
        tau_i_init: float = 20.0,
        tau_mode: str = 'per_pop',
        tau_min: float = 5.0,
        tau_max: float = 200.0,
        dt: float = 25.0,
        noise_scale: float = 0.01,
        spectral_radius: float = 0.9,
        device: str = 'cpu'
    ):
        super().__init__(
            n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, n_outputs=n_outputs,
            tau=tau_e_init, dt=dt, noise_scale=noise_scale,
            spectral_radius=spectral_radius, device=device
        )
        
        self.tau_mode = tau_mode
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        if tau_mode == 'per_pop':
            self.tau_e_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_transform(tau_e_init)))
            self.tau_i_raw = nn.Parameter(torch.tensor(self._inv_sigmoid_transform(tau_i_init)))
        elif tau_mode == 'per_neuron':
            tau_init = torch.full((self.n_total,), tau_e_init)
            tau_init[n_exc:] = tau_i_init
            self.tau_raw = nn.Parameter(self._inv_sigmoid_transform(tau_init))
        else:
            raise ValueError(f"Unknown tau_mode: {tau_mode}")
        
        self.to(device)
    
    def _inv_sigmoid_transform(self, tau):
        """Inverse of sigmoid transform to initialize raw params."""
        x = (tau - self.tau_min) / (self.tau_max - self.tau_min)
        x = torch.clamp(torch.as_tensor(x), 0.01, 0.99)
        return torch.log(x / (1 - x))
    
    def get_tau(self) -> torch.Tensor:
        """Get constrained tau values in [tau_min, tau_max]."""
        if self.tau_mode == 'per_pop':
            tau_e = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_e_raw)
            tau_i = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_i_raw)
            tau = torch.zeros(self.n_total, device=tau_e.device)
            tau[:self.n_exc] = tau_e
            tau[self.n_exc:] = tau_i
            return tau
        else:
            return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_raw)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        initial_state: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with learnable tau."""
        batch_size, n_steps, _ = inputs.shape
        device = inputs.device
        
        if initial_state is None:
            x = torch.zeros(batch_size, self.n_total, device=device)
        else:
            x = initial_state
        
        W_rec = self.W_rec
        tau = self.get_tau()
        alpha = self.dt / tau
        
        rates_list = []
        outputs_list = []
        states_list = [] if return_states else None
        
        for t in range(n_steps):
            r = torch.nn.functional.softplus(x)
            rec_input = torch.matmul(r, W_rec.T)
            ext_input = torch.matmul(inputs[:, t, :], self.W_in.T)
            noise = self.noise_scale * torch.randn_like(x) * (alpha.mean() ** 0.5)
            
            x = (1 - alpha) * x + alpha * (rec_input + ext_input) + noise
            
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


class EIRNN_EyeMLP(EIRNN):
    """EIRNN with separate MLP pathway for eye position inputs."""
    
    def __init__(
        self,
        n_exc: int,
        n_inh: int,
        n_inputs: int,
        n_outputs: int = 2,
        eye_mlp_hidden: int = 16,
        tau: float = 50.0,
        dt: float = 25.0,
        noise_scale: float = 0.01,
        spectral_radius: float = 0.9,
        device: str = 'cpu'
    ):
        n_other_inputs = n_inputs - 2
        n_processed_inputs = n_other_inputs + eye_mlp_hidden
        
        super().__init__(
            n_exc=n_exc, n_inh=n_inh, n_inputs=n_processed_inputs, n_outputs=n_outputs,
            tau=tau, dt=dt, noise_scale=noise_scale,
            spectral_radius=spectral_radius, device=device
        )
        
        self.eye_mlp_hidden = eye_mlp_hidden
        self.original_n_inputs = n_inputs
        
        self.eye_mlp = nn.Sequential(
            nn.Linear(2, eye_mlp_hidden),
            nn.ReLU(),
            nn.Linear(eye_mlp_hidden, eye_mlp_hidden),
            nn.ReLU(),
        )
        
        self.to(device)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        initial_state: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with eye MLP preprocessing."""
        task_inputs = inputs[..., :7]
        eye_inputs = inputs[..., 7:9]
        stim_inputs = inputs[..., 9:]
        
        batch_size, n_steps, _ = eye_inputs.shape
        eye_flat = eye_inputs.reshape(-1, 2)
        eye_processed = self.eye_mlp(eye_flat)
        eye_processed = eye_processed.reshape(batch_size, n_steps, -1)
        
        processed_inputs = torch.cat([task_inputs, eye_processed, stim_inputs], dim=-1)
        
        return super().forward(processed_inputs, initial_state, return_states)


class EIRNN_LowRank(EIRNN):
    """EIRNN with low-rank recurrent connectivity constraint."""
    
    def __init__(
        self,
        n_exc: int,
        n_inh: int,
        n_inputs: int,
        n_outputs: int = 2,
        rank: int = 10,
        tau: float = 50.0,
        dt: float = 25.0,
        noise_scale: float = 0.01,
        device: str = 'cpu'
    ):
        super().__init__(
            n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, n_outputs=n_outputs,
            tau=tau, dt=dt, noise_scale=noise_scale,
            spectral_radius=0.9, device=device
        )
        
        self.rank = rank
        
        # Replace W_rec_raw with low-rank factors
        del self.W_rec_raw
        
        n_total = n_exc + n_inh
        self.U = nn.Parameter(torch.randn(n_total, rank) * 0.1)
        self.V = nn.Parameter(torch.randn(n_total, rank) * 0.1)
        
        self.to(device)
    
    @property
    def W_rec(self) -> torch.Tensor:
        """Recurrent weights reconstructed from low-rank factors with Dale's law."""
        W_raw = torch.matmul(self.U, self.V.T)
        W = torch.abs(W_raw) * self.sign_mask * self.diag_mask
        return W


def create_model_from_data_extended(
    n_classic: int,
    n_interneuron: int,
    n_inputs: int,
    model_type: str = 'standard',
    enforce_ratio: bool = True,
    target_ratio: float = 4.0,
    **kwargs
) -> EIRNN:
    """
    Extended model factory supporting new model variants.
    
    Args:
        model_type: One of 'standard', 'learnable_tau', 'eye_mlp', 'low_rank'
    """
    if enforce_ratio:
        current_ratio = n_classic / max(n_interneuron, 1)
        
        if current_ratio >= target_ratio:
            n_exc = n_classic
            n_inh = int(np.ceil(n_classic / target_ratio))
            n_inh = max(n_inh, n_interneuron)
        else:
            n_inh = n_interneuron
            n_exc = int(np.ceil(n_interneuron * target_ratio))
            n_exc = max(n_exc, n_classic)
    else:
        n_exc = n_classic
        n_inh = n_interneuron
    
    print(f"Model configuration ({model_type}):")
    print(f"  Total: {n_exc} E + {n_inh} I = {n_exc + n_inh}")
    print(f"  E:I ratio: {n_exc/n_inh:.2f}")
    
    if model_type == 'standard':
        return EIRNN(n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, **kwargs)
    elif model_type == 'learnable_tau':
        return EIRNN_LearnableTau(n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, **kwargs)
    elif model_type == 'eye_mlp':
        return EIRNN_EyeMLP(n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, **kwargs)
    elif model_type == 'low_rank':
        return EIRNN_LowRank(n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
