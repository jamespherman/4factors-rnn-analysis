# Fit Quality Improvement Plan for Input-Restructured RNN

## Executive Summary

The input restructuring experiments achieved only ~0.16 validation correlation, while the best model configuration (with attention embedding + learnable h0) achieved ~0.41. The critical issue is that **the model improvements developed in Phases 3-5 were never integrated into the model code** - they exist only in specs and individual training scripts.

**Solution**: Implement attention embedding and learnable h0 in `src/model.py`, then apply these to the input restructuring experiments.

## Problem Analysis

### Current State
| Configuration | Val Correlation | Notes |
|--------------|-----------------|-------|
| Input restructuring E0 baseline | 0.165 | Basic model |
| Input restructuring E2 primary | 0.158 | Basic model, 96% of E0 |
| Best historical model | **0.407** | Attention + learnable h0 |

### Gap Analysis
- Input restructuring uses basic EIRNN without:
  - Attention-based input embedding (contributed +2.1% → ~0.38)
  - Learnable initial state h0 (contributed +4.6% → ~0.40)
  - Extended training (contributed +2.5% → ~0.41)

### Root Cause
The `src/model.py` file does not implement:
1. `InputEmbedding` class with attention variant
2. Learnable h0 (initial hidden state)
3. Support in `create_model_from_data()` for these features

## Implementation Plan

### Phase 1: Implement Missing Model Features

#### 1.1 Add InputEmbedding Class to `src/model.py`

```python
class InputEmbedding(nn.Module):
    """
    Learnable input embedding with multiple variants.

    Types:
    - 'linear': Simple learnable expansion (14 → embed_dim)
    - 'attention': Self-attention over input features (best performer)
    - 'deep': Multi-layer MLP (14 → 28 → embed_dim)
    """

    def __init__(
        self,
        n_inputs: int,
        embed_dim: int,
        embed_type: str = 'attention',
        attention_heads: int = 4
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.embed_dim = embed_dim
        self.embed_type = embed_type
        self.attention_heads = attention_heads

        if embed_type == 'linear':
            self.embed = nn.Linear(n_inputs, embed_dim)
        elif embed_type == 'attention':
            # Self-attention: treat each input as a "token"
            self.qkv_proj = nn.Linear(1, 3 * attention_heads, bias=False)
            self.out_proj = nn.Linear(n_inputs * attention_heads, embed_dim)
        elif embed_type == 'deep':
            hidden = (n_inputs + embed_dim) // 2
            self.embed = nn.Sequential(
                nn.Linear(n_inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed_dim)
            )
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")

    def forward(self, x):
        # x: [batch, time, n_inputs]
        if self.embed_type == 'linear':
            return self.embed(x)
        elif self.embed_type == 'attention':
            batch, time, n_inp = x.shape
            # Reshape to treat each input as a token: [batch*time, n_inputs, 1]
            x_tokens = x.reshape(-1, n_inp, 1)
            # QKV projection: [batch*time, n_inputs, 3*heads]
            qkv = self.qkv_proj(x_tokens)
            # Split and compute attention
            q, k, v = qkv.chunk(3, dim=-1)  # Each [batch*time, n_inputs, heads]
            attn = torch.softmax(q @ k.transpose(-2, -1) / (self.attention_heads ** 0.5), dim=-1)
            out = (attn @ v).reshape(batch * time, -1)  # [batch*time, n_inputs*heads]
            return self.out_proj(out).reshape(batch, time, -1)
        elif self.embed_type == 'deep':
            return self.embed(x)
```

#### 1.2 Modify EIRNN Class to Support Learnable h0 and Input Embedding

Add to `__init__`:
```python
def __init__(
    self,
    ...existing params...,
    input_embed_dim: Optional[int] = None,
    input_embed_type: str = 'attention',
    attention_heads: int = 4,
    learnable_h0: bool = False,
    h0_init: float = 0.1
):
    # ... existing init code ...

    # Input embedding
    if input_embed_dim is not None:
        self.input_embed = InputEmbedding(n_inputs, input_embed_dim,
                                          input_embed_type, attention_heads)
        self.actual_n_inputs = input_embed_dim
    else:
        self.input_embed = None
        self.actual_n_inputs = n_inputs

    # Now create W_in with actual_n_inputs
    self.W_in = nn.Parameter(torch.zeros(self.n_total, self.actual_n_inputs))

    # Learnable initial state
    if learnable_h0:
        self.h0 = nn.Parameter(torch.full((self.n_total,), h0_init))
    else:
        self.h0 = None
```

Modify `forward`:
```python
def forward(self, inputs, ...):
    # Apply input embedding if present
    if self.input_embed is not None:
        inputs = self.input_embed(inputs)

    # Initialize state (use learnable h0 if available)
    if initial_state is None:
        if self.h0 is not None:
            x = self.h0.unsqueeze(0).expand(batch_size, -1)
        else:
            x = torch.zeros(batch_size, self.n_total, device=device)
    else:
        x = initial_state

    # ... rest of forward pass unchanged ...
```

#### 1.3 Update create_model_from_data()

```python
def create_model_from_data(
    n_classic: int,
    n_interneuron: int,
    n_inputs: int,
    enforce_ratio: bool = True,
    target_ratio: float = 4.0,
    input_embed_dim: Optional[int] = None,
    input_embed_type: str = 'attention',
    attention_heads: int = 4,
    learnable_h0: bool = False,
    h0_init: float = 0.1,
    **kwargs
) -> EIRNN:
    # ... existing E:I ratio logic ...

    return EIRNN(
        n_exc=n_exc,
        n_inh=n_inh,
        n_inputs=n_inputs,
        input_embed_dim=input_embed_dim,
        input_embed_type=input_embed_type,
        attention_heads=attention_heads,
        learnable_h0=learnable_h0,
        h0_init=h0_init,
        **kwargs
    )
```

### Phase 2: Update Input Restructuring Training Script

Modify `scripts/run_input_restructuring.py` DEFAULT_CONFIG:

```python
DEFAULT_CONFIG = {
    # Training parameters
    'max_epochs': 2000,  # Extended training
    'patience': 300,     # More patience
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,

    # Model parameters (BEST PHASE 5 CONFIG)
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,
    'input_embed_dim': 56,           # NEW
    'input_embed_type': 'attention', # NEW
    'attention_heads': 4,            # NEW
    'learnable_h0': True,            # NEW
    'h0_init': 0.1,                  # NEW

    # Loss parameters
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'lambda_reg': 1e-4,

    # ... rest unchanged ...
}
```

Update model creation in `train_single_experiment()`:
```python
model = create_model_from_data(
    n_classic=n_exc,
    n_interneuron=n_inh,
    n_inputs=model_n_inputs,
    enforce_ratio=True,
    dt=config['dt'],
    tau=config['tau'],
    noise_scale=config['noise_scale'],
    spectral_radius=config['spectral_radius'],
    input_embed_dim=config.get('input_embed_dim'),
    input_embed_type=config.get('input_embed_type', 'attention'),
    attention_heads=config.get('attention_heads', 4),
    learnable_h0=config.get('learnable_h0', False),
    h0_init=config.get('h0_init', 0.1),
    device=device
)
```

### Phase 3: Run Experiments

#### 3.1 Priority Experiments

1. **Baseline re-run with improvements**: E0 with attention + learnable h0
2. **Primary test**: E2 with improvements (5 seeds)
3. **Comparison**: Verify E2 still achieves ≥80% of improved E0

#### 3.2 Experiment Matrix

| Experiment | Configuration | Seeds | Purpose |
|------------|--------------|-------|---------|
| E0_improved | Full inputs + attention + h0 | 1 | New baseline |
| E2_improved | Input restructuring + attention + h0 | 5 | Primary test |
| E3b_improved | p=0.75 sparsity + improvements | 3 | Optimal sparsity |

### Phase 4: Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| E0 validation correlation | ≥0.35 | ≥0.40 |
| E2 validation correlation | ≥0.28 | ≥0.32 |
| E2/E0 ratio | ≥80% | ≥90% |
| Emergent selectivity preserved | Yes | Yes |

## Implementation Order

1. **Implement `InputEmbedding` class** in `src/model.py`
2. **Add learnable h0 support** to EIRNN in `src/model.py`
3. **Update `create_model_from_data()`** to support new parameters
4. **Update `run_input_restructuring.py`** DEFAULT_CONFIG
5. **Run E0_improved** to establish new baseline
6. **Run E2_improved** (5 seeds) to test input restructuring
7. **Analyze results** and verify emergent selectivity preserved

## Risk Mitigation

1. **If attention doesn't help with restructured inputs**:
   - The restructured inputs may have different characteristics
   - Try linear embedding as fallback
   - Increase embed_dim to 84 (6x)

2. **If fit quality still poor**:
   - Check if neuron-specific inputs are too sparse
   - Try relaxing sparsity constraint (p=0.75)
   - Consider per-neuron embeddings

3. **If emergent selectivity disappears**:
   - Verify input masks are still applied correctly
   - Check that I neurons still don't receive spatial factors
   - This would indicate the improvements somehow bypass the restructuring

## Expected Outcomes

Based on historical data:
- Attention embedding: +2.1% improvement
- Learnable h0: +4.6% improvement
- Extended training: +2.5% improvement

Combined improvement potential: ~10% (additive would give 9.2%)

**Expected E0 with improvements**: 0.165 × 1.10 ≈ **0.18** (conservative) to **0.35-0.40** (if improvements are multiplicative with base performance)

The key uncertainty is whether these improvements scale proportionally with the base performance or are additive. Historical data suggests the latter, meaning:
- Current E0: 0.165
- With attention: 0.165 + (0.02) ≈ 0.185
- With h0: 0.185 + (0.02) ≈ 0.205
- Extended: 0.205 + (0.01) ≈ 0.215

However, if the low baseline is due to missing these features entirely, we might see:
- E0 improving from 0.165 → 0.35+
- E2 improving from 0.158 → 0.28+

This plan will determine which scenario applies.
