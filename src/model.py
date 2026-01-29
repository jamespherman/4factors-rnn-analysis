"""
Excitatory-Inhibitory RNN with Dale's Law constraints.

Based on Song et al. (2016) and Sourmpis et al. (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class InputEmbedding(nn.Module):
    """
    Input embedding layer for expanding input dimensionality.

    Based on Cover's theorem: complex patterns are more likely to be
    linearly separable in higher-dimensional spaces.
    """

    def __init__(
        self,
        n_inputs: int,
        embed_dim: int,
        embed_type: str = 'learnable',
        time_lags: Optional[list] = None,
        embed_hidden_dim: Optional[int] = None,
        embed_n_layers: int = 2,
        input_groups: Optional[list] = None,
        group_embed_dims: Optional[list] = None,
        gru_hidden_dim: Optional[int] = None,
        attention_heads: int = 4
    ):
        """
        Initialize input embedding.

        Args:
            n_inputs: Original input dimension
            embed_dim: Target embedding dimension (ignored for time_lag type)
            embed_type: Type of embedding:
                - 'learnable': Learnable linear + ReLU
                - 'random': Fixed random projection + ReLU
                - 'time_lag': Time-lagged copies of inputs
                - 'deep': Multi-layer MLP (14 -> hidden -> output with ReLU)
                - 'typed': Separate embeddings per input group, concatenated
                - 'recurrent': GRU preprocessing + linear projection
                - 'attention': Self-attention over input features
            time_lags: List of lag values for time_lag type (default [1, 2, 3])
            embed_hidden_dim: Hidden dimension for deep embedding
            embed_n_layers: Number of layers for deep embedding (default 2)
            input_groups: List of (start, end) tuples for typed embedding
            group_embed_dims: Embedding dims per group for typed embedding
            gru_hidden_dim: Hidden dim for recurrent embedding
            attention_heads: Number of attention heads (default 4)
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.embed_type = embed_type
        self.time_lags = time_lags or [1, 2, 3]

        if embed_type == 'learnable':
            self.linear = nn.Linear(n_inputs, embed_dim)
            self.output_dim = embed_dim
        elif embed_type == 'random':
            # Fixed random projection matrix
            W = torch.randn(n_inputs, embed_dim) / np.sqrt(n_inputs)
            self.register_buffer('W', W)
            self.output_dim = embed_dim
        elif embed_type == 'time_lag':
            # Output dim = n_inputs * (1 + len(time_lags))
            self.output_dim = n_inputs * (1 + len(self.time_lags))
        elif embed_type == 'deep':
            # Multi-layer MLP: n_inputs -> hidden -> ... -> embed_dim
            hidden_dim = embed_hidden_dim or (n_inputs + embed_dim) // 2
            layers = []
            in_dim = n_inputs
            for i in range(embed_n_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, embed_dim))
            layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*layers)
            self.output_dim = embed_dim
        elif embed_type == 'typed':
            # Separate embeddings per input group
            if input_groups is None:
                # Default: split into 2 equal groups
                mid = n_inputs // 2
                input_groups = [(0, mid), (mid, n_inputs)]
            if group_embed_dims is None:
                # Default: equal split of embed_dim
                per_group = embed_dim // len(input_groups)
                group_embed_dims = [per_group] * len(input_groups)
                # Handle remainder
                group_embed_dims[-1] = embed_dim - sum(group_embed_dims[:-1])

            self.input_groups = input_groups
            self.group_embed_dims = group_embed_dims
            self.group_linears = nn.ModuleList([
                nn.Linear(end - start, dim)
                for (start, end), dim in zip(input_groups, group_embed_dims)
            ])
            self.output_dim = sum(group_embed_dims)
        elif embed_type == 'recurrent':
            # GRU preprocessing + linear projection
            hidden_dim = gru_hidden_dim or embed_dim
            self.gru = nn.GRU(n_inputs, hidden_dim, batch_first=True, bidirectional=False)
            self.proj = nn.Linear(hidden_dim, embed_dim)
            self.output_dim = embed_dim
        elif embed_type == 'attention':
            # Self-attention over input features (treat each feature as a token)
            # Then project back to embed_dim
            self.attention_heads = attention_heads
            # We'll treat the input as a sequence of 1-d tokens
            # Use a simple query-key-value projection
            self.qkv_proj = nn.Linear(1, 3 * attention_heads)
            self.out_proj = nn.Linear(n_inputs * attention_heads, embed_dim)
            self.output_dim = embed_dim
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply input embedding.

        Args:
            x: [batch, time, n_inputs]

        Returns:
            embedded: [batch, time, output_dim]
        """
        if self.embed_type == 'learnable':
            return F.relu(self.linear(x))
        elif self.embed_type == 'random':
            return F.relu(x @ self.W)
        elif self.embed_type == 'time_lag':
            features = [x]
            for lag in self.time_lags:
                lagged = torch.zeros_like(x)
                if lag < x.shape[1]:
                    lagged[:, lag:, :] = x[:, :-lag, :]
                features.append(lagged)
            return torch.cat(features, dim=-1)
        elif self.embed_type == 'deep':
            return self.mlp(x)
        elif self.embed_type == 'typed':
            # Apply separate embeddings to each group and concatenate
            outputs = []
            for (start, end), linear in zip(self.input_groups, self.group_linears):
                group_input = x[:, :, start:end]
                outputs.append(F.relu(linear(group_input)))
            return torch.cat(outputs, dim=-1)
        elif self.embed_type == 'recurrent':
            # Apply GRU then project
            batch_size, n_steps, _ = x.shape
            gru_out, _ = self.gru(x)  # [batch, time, hidden]
            return F.relu(self.proj(gru_out))
        elif self.embed_type == 'attention':
            # Self-attention over features
            batch_size, n_steps, n_inputs = x.shape
            # Reshape: [batch * time, n_inputs, 1]
            x_flat = x.view(batch_size * n_steps, n_inputs, 1)
            # QKV projection: [batch * time, n_inputs, 3 * heads]
            qkv = self.qkv_proj(x_flat)
            q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch * time, n_inputs, heads]

            # Scaled dot-product attention
            scale = (self.attention_heads ** -0.5)
            attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale  # [batch * time, n_inputs, n_inputs]
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.bmm(attn_weights, v)  # [batch * time, n_inputs, heads]

            # Flatten and project
            attn_out = attn_out.view(batch_size * n_steps, -1)  # [batch * time, n_inputs * heads]
            out = F.relu(self.out_proj(attn_out))  # [batch * time, embed_dim]
            return out.view(batch_size, n_steps, -1)
        return x


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

    where α = dt/τ (or learnable alpha directly)
    """

    # Constants for soft clamping tau to [TAU_MIN, TAU_MAX] ms
    # TAU_MIN must be >= dt to ensure alpha = dt/tau < 1
    TAU_MIN = 25.0  # Changed from 10.0 - must be >= dt to prevent alpha > 1
    TAU_MAX = 200.0

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
        learnable_tau: str = 'none',
        tau_e_init: float = 50.0,
        tau_i_init: float = 35.0,
        learnable_alpha: str = 'none',
        alpha_init: float = 0.5,
        alpha_e_init: Optional[float] = None,
        alpha_i_init: Optional[float] = None,
        low_rank: Optional[int] = None,
        input_embed_dim: Optional[int] = None,
        input_embed_type: str = 'learnable',
        input_time_lags: Optional[list] = None,
        embed_hidden_dim: Optional[int] = None,
        embed_n_layers: int = 2,
        input_groups: Optional[list] = None,
        group_embed_dims: Optional[list] = None,
        gru_hidden_dim: Optional[int] = None,
        attention_heads: int = 4,
        learnable_h0: bool = False,
        h0_init: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize E-I RNN.

        Args:
            n_exc: Number of excitatory units
            n_inh: Number of inhibitory units
            n_inputs: Dimension of input signals
            n_outputs: Dimension of output (default 2 for eye x,y)
            tau: Time constant in ms (used if learnable_tau='none' and learnable_alpha='none')
            dt: Integration timestep in ms (should match data bin size)
            noise_scale: Standard deviation of noise (default 0.1 for better PSTH fitting)
            spectral_radius: Target spectral radius for weight initialization
            bypass_dale: If True, disable Dale's law constraints (for debugging)
            learnable_tau: Time constant learning mode:
                - 'none': Fixed tau for all neurons (default)
                - 'population': Learnable tau_e and tau_i
                - 'neuron': Learnable tau per neuron
            tau_e_init: Initial tau for excitatory neurons (if learnable)
            tau_i_init: Initial tau for inhibitory neurons (if learnable, default 35 to be > dt)
            learnable_alpha: Direct alpha learning mode (alternative to learnable_tau):
                - 'none': Use tau-based alpha (default)
                - 'scalar': Single learnable alpha for all neurons
                - 'population': Separate learnable alpha for E and I
                - 'neuron': Learnable alpha per neuron
            alpha_init: Initial alpha value (used if learnable_alpha != 'none' and
                        alpha_e_init/alpha_i_init not specified)
            alpha_e_init: Initial alpha for E neurons (overrides alpha_init for E)
            alpha_i_init: Initial alpha for I neurons (overrides alpha_init for I)
            low_rank: If specified, constrain W_rec to this rank (W_rec = U @ V^T)
            input_embed_dim: If specified, expand inputs to this dimension
            input_embed_type: Type of input embedding ('learnable', 'random', 'time_lag',
                            'deep', 'typed', 'recurrent', 'attention')
            input_time_lags: List of time lags for 'time_lag' embedding type
            embed_hidden_dim: Hidden dimension for deep embedding
            embed_n_layers: Number of layers for deep embedding (default 2)
            input_groups: List of (start, end) tuples for typed embedding
            group_embed_dims: Embedding dims per group for typed embedding
            gru_hidden_dim: Hidden dim for recurrent embedding
            attention_heads: Number of attention heads (default 4)
            learnable_h0: If True, learn initial hidden state (per-neuron)
            h0_init: Initial value for learnable h0 (default 0.1)
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = dt
        self.noise_scale = noise_scale
        self.spectral_radius = spectral_radius
        self.bypass_dale = bypass_dale
        self.learnable_tau = learnable_tau
        self.learnable_alpha = learnable_alpha
        self.low_rank = low_rank

        # === Input embedding (Cover's theorem) ===
        # Enable embedding for any non-default type or explicit embed_dim
        needs_embedding = (
            input_embed_dim is not None or
            input_embed_type in ['time_lag', 'deep', 'typed', 'recurrent', 'attention']
        )
        if needs_embedding:
            self.input_embed = InputEmbedding(
                n_inputs=n_inputs,
                embed_dim=input_embed_dim or n_inputs,
                embed_type=input_embed_type,
                time_lags=input_time_lags,
                embed_hidden_dim=embed_hidden_dim,
                embed_n_layers=embed_n_layers,
                input_groups=input_groups,
                group_embed_dims=group_embed_dims,
                gru_hidden_dim=gru_hidden_dim,
                attention_heads=attention_heads
            )
            actual_n_inputs = self.input_embed.output_dim
        else:
            self.input_embed = None
            actual_n_inputs = n_inputs

        # Store actual input dim for W_in
        self.actual_n_inputs = actual_n_inputs

        # Validate: can't use both learnable_tau and learnable_alpha
        if learnable_tau != 'none' and learnable_alpha != 'none':
            raise ValueError("Cannot use both learnable_tau and learnable_alpha. Choose one.")

        # Determine alpha initialization values
        # Use separate alpha_e_init/alpha_i_init if provided
        if alpha_e_init is None:
            alpha_e_init = alpha_init
        if alpha_i_init is None:
            # Default: I neurons slightly faster than E
            alpha_i_init = min(0.95, alpha_init * 1.5)

        # === Alpha configuration (integration constant) ===
        if learnable_alpha != 'none':
            # Direct alpha learning (more stable than tau learning)
            self.tau = None
            self._tau_e = None
            self._tau_i = None
            self._tau_per_neuron = None
            self._init_learnable_alpha(learnable_alpha, alpha_e_init, alpha_i_init)
        elif learnable_tau != 'none':
            # Tau-based alpha with soft clamping
            self._alpha_logit = None
            self._alpha_e_logit = None
            self._alpha_i_logit = None
            self._init_learnable_tau(learnable_tau, tau_e_init, tau_i_init)
        else:
            # Fixed tau/alpha
            self.tau = tau
            self.alpha = dt / tau
            self._tau_e = None
            self._tau_i = None
            self._tau_per_neuron = None
            self._alpha_logit = None
            self._alpha_e_logit = None
            self._alpha_i_logit = None

        # === Recurrent weight configuration ===
        if low_rank is not None:
            # Low-rank factorization: W_rec = U @ V^T
            self.U = nn.Parameter(torch.randn(self.n_total, low_rank) / np.sqrt(low_rank))
            self.V = nn.Parameter(torch.randn(self.n_total, low_rank) / np.sqrt(low_rank))
            self.W_rec_raw = None  # Not used for low-rank
        else:
            # Full-rank recurrent weights
            self.W_rec_raw = nn.Parameter(torch.zeros(self.n_total, self.n_total))
            self.U = None
            self.V = None

        # Other learnable parameters
        self.W_in = nn.Parameter(torch.zeros(self.n_total, actual_n_inputs))
        self.W_out = nn.Parameter(torch.zeros(n_outputs, n_exc))  # Only E neurons
        self.b_out = nn.Parameter(torch.zeros(n_outputs))

        # Learnable rate scaling (initialize near target mean firing rate)
        # This addresses the scale mismatch: softplus outputs ~1 sp/s, target is ~11 sp/s
        self.rate_scale = nn.Parameter(torch.tensor(10.0))
        self.rate_baseline = nn.Parameter(torch.tensor(0.5))

        # Learnable initial hidden state (per-neuron)
        self.learnable_h0 = learnable_h0
        if learnable_h0:
            self.h0 = nn.Parameter(torch.full((self.n_total,), h0_init))
        else:
            self.h0 = None

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

    def _init_learnable_alpha(self, mode: str, alpha_e_init: float, alpha_i_init: float):
        """Initialize learnable alpha parameters with logit parameterization."""
        if mode == 'scalar':
            # Single learnable alpha for entire network (use E init as baseline)
            self._alpha_logit = nn.Parameter(torch.tensor(self._alpha_to_logit(alpha_e_init)))
            self._alpha_e_logit = None
            self._alpha_i_logit = None
        elif mode == 'population':
            # Separate alpha for E and I populations
            self._alpha_e_logit = nn.Parameter(torch.tensor(self._alpha_to_logit(alpha_e_init)))
            self._alpha_i_logit = nn.Parameter(torch.tensor(self._alpha_to_logit(alpha_i_init)))
            self._alpha_logit = None
        elif mode == 'neuron':
            # Per-neuron alpha
            alpha_logit_e = self._alpha_to_logit(alpha_e_init)
            alpha_logit_i = self._alpha_to_logit(alpha_i_init)
            alpha_init_tensor = torch.cat([
                torch.full((self.n_exc,), alpha_logit_e),
                torch.full((self.n_inh,), alpha_logit_i)
            ])
            self._alpha_logit = nn.Parameter(alpha_init_tensor)
            self._alpha_e_logit = None
            self._alpha_i_logit = None
        else:
            raise ValueError(f"Unknown learnable_alpha mode: {mode}")

    def _init_learnable_tau(self, mode: str, tau_e_init: float, tau_i_init: float):
        """Initialize learnable tau parameters with soft clamping via softplus."""
        # Use softplus parameterization: tau = TAU_MIN + softplus(raw_tau) * scale
        # where scale = (TAU_MAX - TAU_MIN) / softplus(0) ≈ (TAU_MAX - TAU_MIN) / 0.693
        # This ensures smooth gradients everywhere
        if mode == 'population':
            self.tau = None
            self._tau_e_raw = nn.Parameter(torch.tensor(self._tau_to_raw(tau_e_init)))
            self._tau_i_raw = nn.Parameter(torch.tensor(self._tau_to_raw(tau_i_init)))
            self._tau_per_neuron_raw = None
            # Keep old names for backward compatibility but don't use them
            self._tau_e = None
            self._tau_i = None
            self._tau_per_neuron = None
        elif mode == 'neuron':
            self.tau = None
            self._tau_e_raw = None
            self._tau_i_raw = None
            tau_init_tensor = torch.cat([
                torch.full((self.n_exc,), self._tau_to_raw(tau_e_init)),
                torch.full((self.n_inh,), self._tau_to_raw(tau_i_init))
            ])
            self._tau_per_neuron_raw = nn.Parameter(tau_init_tensor)
            self._tau_e = None
            self._tau_i = None
            self._tau_per_neuron = None
        else:
            raise ValueError(f"Unknown learnable_tau mode: {mode}")

    @staticmethod
    def _alpha_to_logit(alpha: float) -> float:
        """Convert alpha in (0, 1) to unconstrained logit space."""
        alpha = np.clip(alpha, 0.01, 0.99)
        return float(np.log(alpha / (1 - alpha)))

    @staticmethod
    def _logit_to_alpha(logit: torch.Tensor) -> torch.Tensor:
        """Convert logit to alpha in (0, 1) via sigmoid."""
        return torch.sigmoid(logit)

    def _tau_to_raw(self, tau: float) -> float:
        """Convert tau to raw parameter space (inverse of soft clamping)."""
        # tau = TAU_MIN + softplus(raw) * scale, where scale maps softplus range to tau range
        # Approximate: raw ≈ inverse_softplus((tau - TAU_MIN) / scale)
        scale = (self.TAU_MAX - self.TAU_MIN) / 2.0  # Scale factor
        x = (tau - self.TAU_MIN) / scale
        x = np.clip(x, 0.01, 10.0)  # Avoid numerical issues
        # inverse softplus: raw = log(exp(x) - 1)
        if x > 10:
            return float(x)  # For large x, softplus(x) ≈ x
        return float(np.log(np.exp(x) - 1 + 1e-8))

    def _raw_to_tau(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw parameter to tau via soft clamping (softplus)."""
        scale = (self.TAU_MAX - self.TAU_MIN) / 2.0
        return self.TAU_MIN + torch.nn.functional.softplus(raw) * scale
    
    def _initialize_weights(self):
        """Initialize weights with balanced E/I and controlled spectral radius."""
        n_exc = self.n_exc
        n_inh = self.n_inh
        n_total = self.n_total

        if self.low_rank is not None:
            # Low-rank initialization: U, V already initialized in __init__
            # Scale to approximate target spectral radius
            # For W = U @ V^T, spectral radius depends on U, V norms
            # We initialize with small values and let training adjust
            scale = self.spectral_radius / np.sqrt(self.low_rank)
            self.U.data *= scale
            self.V.data *= scale
        elif self.bypass_dale:
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
        if self.low_rank is not None:
            # Low-rank: W_rec = U @ V^T
            W_low_rank = self.U @ self.V.T

            if self.bypass_dale:
                W = W_low_rank * self.diag_mask
            else:
                # Apply Dale's law: take absolute value, then apply sign mask
                W = torch.abs(W_low_rank) * self.sign_mask * self.diag_mask
        elif self.bypass_dale:
            # No sign constraints - weights can be positive or negative
            W = self.W_rec_raw * self.diag_mask
        else:
            # Apply Dale's law: |W| * sign_mask * diag_mask
            W = torch.abs(self.W_rec_raw) * self.sign_mask * self.diag_mask
        return W

    def get_alpha(self, device: torch.device) -> torch.Tensor:
        """
        Get integration constants (alpha = dt/tau) for all neurons.

        Returns:
            alpha: scalar or [n_total] tensor of integration constants
        """
        # Case 1: Direct learnable alpha
        if self.learnable_alpha != 'none':
            if self.learnable_alpha == 'scalar':
                return self._logit_to_alpha(self._alpha_logit)
            elif self.learnable_alpha == 'population':
                alpha_e = self._logit_to_alpha(self._alpha_e_logit)
                alpha_i = self._logit_to_alpha(self._alpha_i_logit)
                return torch.cat([
                    alpha_e.expand(self.n_exc),
                    alpha_i.expand(self.n_inh)
                ])
            elif self.learnable_alpha == 'neuron':
                return self._logit_to_alpha(self._alpha_logit)

        # Case 2: Learnable tau (compute alpha = dt/tau)
        if self.learnable_tau == 'population':
            # Soft clamping via softplus
            tau_e = self._raw_to_tau(self._tau_e_raw)
            tau_i = self._raw_to_tau(self._tau_i_raw)
            alpha_e = self.dt / tau_e
            alpha_i = self.dt / tau_i
            return torch.cat([
                alpha_e.expand(self.n_exc),
                alpha_i.expand(self.n_inh)
            ])

        elif self.learnable_tau == 'neuron':
            # Soft clamping via softplus
            tau = self._raw_to_tau(self._tau_per_neuron_raw)
            return self.dt / tau

        # Case 3: Fixed tau/alpha
        return torch.tensor(self.alpha, device=device)

    def get_tau_values(self) -> dict:
        """Get current tau values for logging/analysis."""
        if self.learnable_tau == 'none':
            if self.tau is not None:
                return {'tau': self.tau}
            else:
                # Using learnable_alpha, no tau
                return {}
        elif self.learnable_tau == 'population':
            tau_e = self._raw_to_tau(self._tau_e_raw)
            tau_i = self._raw_to_tau(self._tau_i_raw)
            return {
                'tau_e': tau_e.item(),
                'tau_i': tau_i.item()
            }
        elif self.learnable_tau == 'neuron':
            tau = self._raw_to_tau(self._tau_per_neuron_raw)
            return {
                'tau_e_mean': tau[:self.n_exc].mean().item(),
                'tau_e_std': tau[:self.n_exc].std().item(),
                'tau_i_mean': tau[self.n_exc:].mean().item(),
                'tau_i_std': tau[self.n_exc:].std().item()
            }
        return {}

    def get_alpha_values(self) -> dict:
        """Get current alpha values for logging/analysis."""
        if self.learnable_alpha == 'none':
            if hasattr(self, 'alpha') and self.alpha is not None:
                return {'alpha': self.alpha}
            else:
                # Compute from tau
                tau_vals = self.get_tau_values()
                if 'tau' in tau_vals:
                    return {'alpha': self.dt / tau_vals['tau']}
                elif 'tau_e' in tau_vals:
                    return {
                        'alpha_e': self.dt / tau_vals['tau_e'],
                        'alpha_i': self.dt / tau_vals['tau_i']
                    }
                return {}
        elif self.learnable_alpha == 'scalar':
            alpha = self._logit_to_alpha(self._alpha_logit)
            return {'alpha': alpha.item()}
        elif self.learnable_alpha == 'population':
            alpha_e = self._logit_to_alpha(self._alpha_e_logit)
            alpha_i = self._logit_to_alpha(self._alpha_i_logit)
            return {
                'alpha_e': alpha_e.item(),
                'alpha_i': alpha_i.item()
            }
        elif self.learnable_alpha == 'neuron':
            alpha = self._logit_to_alpha(self._alpha_logit)
            return {
                'alpha_e_mean': alpha[:self.n_exc].mean().item(),
                'alpha_e_std': alpha[:self.n_exc].std().item(),
                'alpha_i_mean': alpha[self.n_exc:].mean().item(),
                'alpha_i_std': alpha[self.n_exc:].std().item()
            }
        return {}
    
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

        # Apply input embedding if configured (Cover's theorem expansion)
        if self.input_embed is not None:
            inputs = self.input_embed(inputs)

        # Initialize state
        if initial_state is not None:
            x = initial_state
        elif self.h0 is not None:
            # Use learnable initial state (expand to batch dimension)
            x = self.h0.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            x = torch.zeros(batch_size, self.n_total, device=device)

        # Get constrained weights
        W_rec = self.W_rec

        # Get alpha values (may be per-neuron if learnable_tau is enabled)
        alpha = self.get_alpha(device)

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
            # Use mean alpha for noise scaling when per-neuron alpha is used
            if alpha.dim() == 0:
                alpha_for_noise = alpha
            else:
                alpha_for_noise = alpha.mean()
            noise = self.noise_scale * torch.randn_like(x) * (alpha_for_noise ** 0.5)

            # State update (Euler integration)
            # alpha can be scalar or [n_total] for per-neuron time constants
            x = (1 - alpha) * x + alpha * (rec_input + ext_input) + noise

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
    learnable_tau: str = 'none',
    tau_e_init: float = 50.0,
    tau_i_init: float = 35.0,
    learnable_alpha: str = 'none',
    alpha_init: float = 0.5,
    alpha_e_init: Optional[float] = None,
    alpha_i_init: Optional[float] = None,
    low_rank: Optional[int] = None,
    input_embed_dim: Optional[int] = None,
    input_embed_type: str = 'learnable',
    input_time_lags: Optional[list] = None,
    embed_hidden_dim: Optional[int] = None,
    embed_n_layers: int = 2,
    input_groups: Optional[list] = None,
    group_embed_dims: Optional[list] = None,
    gru_hidden_dim: Optional[int] = None,
    attention_heads: int = 4,
    learnable_h0: bool = False,
    h0_init: float = 0.1,
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
        learnable_tau: Time constant learning mode ('none', 'population', 'neuron')
        tau_e_init: Initial tau for excitatory neurons (if learnable)
        tau_i_init: Initial tau for inhibitory neurons (if learnable, default 35ms > dt)
        learnable_alpha: Direct alpha learning mode ('none', 'scalar', 'population', 'neuron')
        alpha_init: Initial alpha value (used if learnable_alpha != 'none')
        alpha_e_init: Initial alpha for E neurons (overrides alpha_init)
        alpha_i_init: Initial alpha for I neurons (overrides alpha_init)
        low_rank: If specified, constrain W_rec to this rank
        input_embed_dim: If specified, expand inputs to this dimension
        input_embed_type: Type of input embedding ('learnable', 'random', 'time_lag',
                        'deep', 'typed', 'recurrent', 'attention')
        input_time_lags: List of time lags for 'time_lag' embedding type
        embed_hidden_dim: Hidden dimension for deep embedding
        embed_n_layers: Number of layers for deep embedding (default 2)
        input_groups: List of (start, end) tuples for typed embedding
        group_embed_dims: Embedding dims per group for typed embedding
        gru_hidden_dim: Hidden dim for recurrent embedding
        attention_heads: Number of attention heads (default 4)
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

    return EIRNN(
        n_exc=n_exc, n_inh=n_inh, n_inputs=n_inputs,
        bypass_dale=bypass_dale,
        learnable_tau=learnable_tau,
        tau_e_init=tau_e_init,
        tau_i_init=tau_i_init,
        learnable_alpha=learnable_alpha,
        alpha_init=alpha_init,
        alpha_e_init=alpha_e_init,
        alpha_i_init=alpha_i_init,
        low_rank=low_rank,
        input_embed_dim=input_embed_dim,
        input_embed_type=input_embed_type,
        input_time_lags=input_time_lags,
        embed_hidden_dim=embed_hidden_dim,
        embed_n_layers=embed_n_layers,
        input_groups=input_groups,
        group_embed_dims=group_embed_dims,
        gru_hidden_dim=gru_hidden_dim,
        attention_heads=attention_heads,
        learnable_h0=learnable_h0,
        h0_init=h0_init,
        **kwargs
    )


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
