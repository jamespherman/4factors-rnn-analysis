"""
Bilateral superior colliculus E-I RNN.

Unit layout (global index order):
  [ L_E (80) | L_I (20) | R_E (80) | R_I (20) ]
Within each hemisphere the E units are split into n_loc_hemi topographic pools
(20 E units per location). The LEFT SC represents the RIGHT hemifield
(locations n_loc_hemi..2*n_loc_hemi-1), the RIGHT SC the LEFT hemifield.
Within each hemisphere, the first frac_inh_commissural of I units are
"commissural" (allowed to project to the other hemisphere); the rest are
local-only.

Connectivity constraints (fixed structural masks):
  - Dale's law: E columns >= 0, I columns <= 0 (W = |W_raw| * sign * mask)
  - within hemisphere: all E/I pairs allowed with probability p_within
  - between hemispheres: ONLY commissural I -> (E, I) with probability p_commissural
  - no self-connections

Dynamics (Euler, as in the Aim 1 RNN):
  x[t] = (1-a) x[t-1] + a (W_rec r[t-1] + W_in u[t] + b) + noise * sqrt(a) * xi
  r[t] = softplus(x[t])

Policy (per time step, softmax over 1 + n_loc actions):
  fixate: learnable linear readout of all E units + bias
  saccade to location k: non-negative readout of the E pool representing k + shared bias
Value baseline (for REINFORCE): linear readout of all units + inputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from task import n_inputs


class UnitIndex:
    """Index bookkeeping for the bilateral layout."""

    def __init__(self, cfg: Config):
        nE, nI, nH = cfg.n_exc_hemi, cfg.n_inh_hemi, cfg.n_hemi
        nIc = int(round(cfg.frac_inh_commissural * nI))
        self.hemi = np.repeat([0, 1], nH)                 # 0 = left SC, 1 = right SC
        self.is_exc = np.tile(np.r_[np.ones(nE, bool), np.zeros(nI, bool)], 2)
        self.is_inh = ~self.is_exc
        comm = np.zeros(cfg.n_total, bool)
        for h in (0, 1):
            comm[h * nH + nE: h * nH + nE + nIc] = True
        self.is_comm_inh = comm
        # location represented by each E unit (-1 for I units)
        loc = -np.ones(cfg.n_total, int)
        per = nE // cfg.n_loc_hemi
        for h in (0, 1):
            # left SC (h=0) -> right hemifield locations; right SC (h=1) -> left hemifield
            base = cfg.n_loc_hemi if h == 0 else 0
            for j in range(cfg.n_loc_hemi):
                loc[h * nH + j * per: h * nH + (j + 1) * per] = base + j
        self.loc = loc
        # hemifield represented by each unit (contralateral): 0 = left field, 1 = right field
        self.field = 1 - self.hemi


def build_masks(cfg: Config, idx: UnitIndex, rng: np.random.Generator):
    N = cfg.n_total
    same = idx.hemi[:, None] == idx.hemi[None, :]      # [post, pre]
    within = same & (rng.random((N, N)) < cfg.p_within)
    cross = (~same) & idx.is_comm_inh[None, :] & (rng.random((N, N)) < cfg.p_commissural)
    if cfg.comm_exc_to_inh:
        cross |= (~same) & idx.is_exc[None, :] & idx.is_inh[:, None] & (rng.random((N, N)) < cfg.p_comm_exc)
    if not cfg.commissural:
        cross[:] = False
    mask = (within | cross).astype(np.float32)
    np.fill_diagonal(mask, 0)
    sign = np.where(idx.is_exc, 1.0, -1.0)[None, :].repeat(N, 0).astype(np.float32)
    return mask, sign, cross.astype(np.float32)


class BilateralEIRNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.idx = UnitIndex(cfg)
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        N, M = cfg.n_total, n_inputs(cfg)
        self.alpha = cfg.dt / cfg.tau

        mask, sign, cross = build_masks(cfg, self.idx, rng)
        self.register_buffer("mask", torch.tensor(mask))
        self.register_buffer("sign", torch.tensor(sign))
        self.register_buffer("cross_mask", torch.tensor(cross))
        self.register_buffer("comm_scale", torch.tensor(1.0))  # set to 0 for post-hoc ablation
        # per-unit output gain (1 = intact, 0 = silenced) for lesion analyses; not saved in checkpoints
        self.register_buffer("unit_gain", torch.ones(N), persistent=False)

        # --- recurrent weights: gamma init, E/I balanced per row, spectral radius ---
        W = rng.gamma(2.0, 0.05, size=(N, N)) * mask
        E, I = self.idx.is_exc, self.idx.is_inh
        e_sum = W[:, E].sum(1, keepdims=True)
        i_sum = W[:, I].sum(1, keepdims=True)
        W[:, I] *= np.where(i_sum > 0, e_sum / (i_sum + 1e-8), 1.0)
        Ws = W * sign
        rad = np.max(np.abs(np.linalg.eigvals(Ws)))
        W *= cfg.spectral_radius / rad
        self.W_rec_raw = nn.Parameter(torch.tensor(W, dtype=torch.float32))

        # --- input weights ---
        # sensory (visual + goal) weights: fixed topographic map onto E pools
        W_sens = np.zeros((N, M), np.float32)
        for k in range(cfg.n_loc):
            pool = (self.idx.loc == k)
            hemi_of_k = self.idx.hemi[pool][0]
            W_sens[pool, 1 + k] = cfg.vis_gain * rng.uniform(0.5, 1.5, pool.sum())
            W_sens[pool, 1 + cfg.n_loc + k] = cfg.goal_gain * rng.uniform(0.5, 1.5, pool.sum())
            inh_h = I & (self.idx.hemi == hemi_of_k)
            if cfg.sensory_to_inh:
                W_sens[inh_h, 1 + k] = cfg.vis_gain * rng.uniform(0.25, 0.75, inh_h.sum())
            if cfg.goal_to_inh:
                W_sens[inh_h, 1 + cfg.n_loc + k] = cfg.goal_gain * rng.uniform(0.25, 0.75, inh_h.sum())
        sens_mask = np.zeros((N, M), np.float32)
        sens_mask[:, 1:1 + 2 * cfg.n_loc] = 1.0
        self.register_buffer("sens_mask", torch.tensor(sens_mask))
        if cfg.learn_sensory_weights:
            # learnable but non-negative, restricted to the same (topographic) support
            self.W_sens_raw = nn.Parameter(torch.tensor(W_sens))
            self.register_buffer("sens_support", torch.tensor((W_sens > 0).astype(np.float32)))
        else:
            self.register_buffer("W_sens_fixed", torch.tensor(W_sens))
        # fixation + reward-context weights: learnable, any sign
        other = np.zeros((N, M), np.float32)
        other[:, 0] = rng.normal(0, 0.5, N)
        self.W_other = nn.Parameter(torch.tensor(other))
        om = np.zeros((N, M), np.float32)
        om[:, 0] = 1
        if cfg.reward_context_input:
            om[:, -2:] = 1
        self.register_buffer("other_mask", torch.tensor(om))
        self.bias = nn.Parameter(torch.zeros(N))

        # --- policy readout ---
        nE_tot = int(E.sum())
        self.exc_idx = torch.tensor(np.where(E)[0])
        pool_mask = np.zeros((cfg.n_loc, N), np.float32)
        for k in range(cfg.n_loc):
            pool_mask[k, self.idx.loc == k] = 1.0
        self.register_buffer("pool_mask", torch.tensor(pool_mask))
        # identical readout of every pool (mean E rate), shared learnable gain + bias:
        # the downstream saccade generator treats all SC sites alike (prevents a
        # trivial always-one-side policy via asymmetric readout weights)
        self.register_buffer("pool_avg", torch.tensor(pool_mask / pool_mask.sum(1, keepdims=True)))
        self.sac_gain_raw = nn.Parameter(torch.tensor(1.0))
        self.b_sac = nn.Parameter(torch.tensor(-5.0))
        W_fix = np.zeros((1, N), np.float32)
        self.W_fix = nn.Parameter(torch.tensor(W_fix))
        self.register_buffer("exc_mask", torch.tensor(E.astype(np.float32))[None, :])
        self.b_fix = nn.Parameter(torch.tensor(cfg.fix_bias_init))
        self.w_fix_in = nn.Parameter(torch.tensor(4.0))   # used when fix_readout == "input"
        # value baseline
        self.W_val = nn.Linear(N + M + 1, 1)
        nn.init.zeros_(self.W_val.weight)
        nn.init.zeros_(self.W_val.bias)

    # ------------------------------------------------------------------
    @property
    def W_rec(self):
        W = torch.abs(self.W_rec_raw) * self.sign * self.mask
        # post-hoc commissural ablation (comm_scale = 0) leaves within-hemisphere weights intact
        return W * (1 - self.cross_mask) + W * self.cross_mask * self.comm_scale

    @property
    def W_in(self):
        if self.cfg.learn_sensory_weights:
            Ws = torch.abs(self.W_sens_raw) * self.sens_support
        else:
            Ws = self.W_sens_fixed
        return Ws + self.W_other * self.other_mask

    def phi(self, x):
        if self.cfg.nonlinearity == "softplus":
            return F.softplus(x)
        return F.relu(x)

    def forward(self, u, noise=True, generator=None, return_x=False):
        """u: [B, T, M] -> rates [B, T, N]."""
        B, T, _ = u.shape
        cfg = self.cfg
        W = self.W_rec
        Win = self.W_in
        ext = u @ Win.T + self.bias  # [B, T, N]
        x = torch.zeros(B, cfg.n_total, device=u.device)
        rates, xs = [], []
        sq = cfg.noise_scale * np.sqrt(self.alpha)
        g = self.unit_gain
        for t in range(T):
            r = self.phi(x) * g
            x = (1 - self.alpha) * x + self.alpha * (r @ W.T + ext[:, t])
            if noise and cfg.noise_scale > 0:
                x = x + sq * torch.randn(x.shape, device=u.device, generator=generator)
            rates.append(self.phi(x) * g)
            if return_x:
                xs.append(x)
        R = torch.stack(rates, 1)
        if return_x:
            return R, torch.stack(xs, 1)
        return R

    def policy_logits(self, R, u=None, urg=None):
        """R: [B, T, N] -> logits [B, T, 1 + n_loc] (action 0 = fixate).
        u: inputs (needed for fix_readout == 'input'); urg: [B, T] urgency ramp."""
        sac = F.softplus(self.sac_gain_raw) * (R @ self.pool_avg.T) + self.b_sac
        if self.cfg.fix_readout == "input":
            fix = (self.b_fix + self.w_fix_in * u[..., :1])
        else:
            fix = R @ (self.W_fix * self.exc_mask).T + self.b_fix
        if urg is not None:
            fix = fix - urg[..., None]
        return torch.cat([fix, sac], -1)

    def value(self, R, u, tfrac):
        return self.W_val(torch.cat([R, u, tfrac], -1)).squeeze(-1)

    def weight_blocks(self):
        """Return W_rec as numpy plus index helper for analyses."""
        return self.W_rec.detach().cpu().numpy(), self.idx
