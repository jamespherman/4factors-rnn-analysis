"""
Bilateral SC + upstream (BG-like) module for the srsClassic hub/resolver comparison (plan v2).

SC (shared by all architectures)
  Layout per side s (0 = LEFT SC, represents location 1 = RIGHT field; 1 = RIGHT SC, represents location 0):
    [E_pool (n_pool) | E_other (n_e_other) | I_local (n_i_local) | I_comm (n_i_comm)]
  Dale's law (W = |W_raw| * sign * mask), no autapses; within-side p_within; cross-side only I_comm -> (E, I) with
  p_comm (if cfg.commissural). Rates softplus, tau 50 ms, dt 10 ms, state noise noise_sc*sqrt(dt/tau).
  Inputs onto the E pool of the SC representing location i: visual (fixed), SNr inhibition (fixed weight w_snr),
  side bias (if bias_locus == "SC"), slow OU noise. Fixation input: learnable weights onto all SC units.
  Architecture A additionally feeds identity-at-location (4) and belief (2) to ALL SC units through FIXED random
  weights, so binding identity x belief x location must happen in SC's own nonlinear recurrent dynamics.

UPSTREAM (same size/trainability in all architectures; no SC -> upstream feedback in v2)
  Recurrent softplus module (no Dale), inputs: fixation, belief, identity-at-location (B and C only; delayed by
  up_latency), side bias (if bias_locus == "upstream"). Output -> disinhibition per location:
    B, A: dis_i = dis_max * sigmoid(o_i)                               (independent, graded)
    C   : dis   = dis_max * softmax(c_beta * [o_L, o_R, o_nogo])[:2]  (upstream WTA, near-categorical)
  SNr_i(t) = relu(snr_tonic + snr_fix * fixation(t) - dis_i(t))      (tonic GABAergic output; inhibits pool i)
  In A the upstream receives no identity input, so its output cannot carry the identity x belief conjunction.

READOUT (no location normalisation)
  hazard_i(t) = sigmoid(g * (mean pool rate for location i - theta)); shared g (softplus) and theta.
  Each step each location fires independently; the first event triggers a saccade there (ties broken at random);
  an event before go is a fixation break.

PERTURBATION BUFFERS (set via simulate.apply_perturbation)
  w_gain [N, N]   multiplies the effective SC weights (e.g. GABA-A blockade of I -> E at one site)
  snr_gain [N]    multiplies SNr -> unit inhibition (blockade of nigral GABA-A at one site)
  unit_gain [N]   multiplies SC unit output (silencing; muscimol)
  dis_override    optional callable(dis, inputs) -> dis (upstream manipulations)
  hand_upstream   optional callable(inputs, noise_fn) -> dis [B, T, 2] for hand-built networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SRSConfig

TYPES = ("E_pool", "E_other", "I_local", "I_comm")


class SCIndex:
    def __init__(self, cfg: SRSConfig):
        nS = cfg.n_side
        counts = (cfg.n_pool, cfg.n_e_other, cfg.n_i_local, cfg.n_i_comm)
        tp = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        self.side = np.repeat([0, 1], nS)
        self.type = np.tile(tp, 2)                       # index into TYPES
        self.is_exc = self.type <= 1
        self.is_inh = ~self.is_exc
        self.is_pool = self.type == 0
        self.is_comm = self.type == 3
        self.field = 1 - self.side                       # location represented by the unit's SC
        self.loc = np.where(self.is_pool, self.field, -1)
        self.n = 2 * nS

    def pool(self, loc):
        return np.where(self.is_pool & (self.field == loc))[0]

    def meta(self):
        return dict(sc_side=np.where(self.side == 0, "L", "R"), type=np.array(TYPES)[self.type],
                    field=np.where(self.field == 0, "L", "R"), is_exc=self.is_exc, pool_loc=self.loc)


class SRSNet(nn.Module):
    def __init__(self, cfg: SRSConfig):
        super().__init__()
        self.cfg = cfg
        self.idx = SCIndex(cfg)
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        N = self.idx.n
        self.alpha = cfg.dt / cfg.tau
        self.alpha_up = cfg.dt / cfg.tau_up

        # ---------------- SC recurrent weights ----------------
        idx = self.idx
        same = idx.side[:, None] == idx.side[None, :]
        within = same & (rng.random((N, N)) < cfg.p_within)
        cross = (~same) & idx.is_comm[None, :] & (rng.random((N, N)) < cfg.p_comm)
        if not cfg.commissural:
            cross[:] = False
        mask = (within | cross).astype(np.float32)
        np.fill_diagonal(mask, 0)
        sign = np.where(idx.is_exc, 1.0, -1.0)[None, :].repeat(N, 0).astype(np.float32)
        W = rng.gamma(2.0, 0.05, (N, N)) * mask
        E, I = idx.is_exc, idx.is_inh
        e_sum, i_sum = W[:, E].sum(1, keepdims=True), W[:, I].sum(1, keepdims=True)
        W[:, I] *= np.where(i_sum > 0, e_sum / (i_sum + 1e-8), 1.0)
        rad = np.max(np.abs(np.linalg.eigvals(W * sign)))
        W *= cfg.spectral_radius / max(rad, 1e-6)
        self.W_raw = nn.Parameter(torch.tensor(W, dtype=torch.float32))
        self.register_buffer("mask", torch.tensor(mask))
        self.register_buffer("sign", torch.tensor(sign))
        self.register_buffer("cross_mask", torch.tensor(cross.astype(np.float32)))
        self.b_sc = nn.Parameter(torch.zeros(N))

        # ---------------- SC input weights ----------------
        pool_of = np.zeros((N, 2), np.float32)            # unit receives location-i pool inputs
        for loc in (0, 1):
            pool_of[idx.pool(loc), loc] = 1.0
        self.register_buffer("pool_of", torch.tensor(pool_of))
        self.register_buffer("w_vis", torch.tensor(pool_of * cfg.w_vis * rng.uniform(0.5, 1.5, (N, 1)).astype(np.float32)))
        self.w_fix_sc = nn.Parameter(torch.zeros(N)) if cfg.learn_sc_fix else None
        if cfg.arch == "A":
            Wa = rng.normal(0, 1, (N, 6)).astype(np.float32) * cfg.a_input_gain
            tgt = {"all": np.ones(N, bool), "E": self.idx.is_exc, "E_other": self.idx.type == 1}[cfg.a_input_targets]
            Wa[~tgt] = 0.0
            self.register_buffer("w_a", torch.tensor(Wa))     # fixed random: identity (4) + belief (2)
        else:
            self.w_a = None

        # ---------------- upstream ----------------
        nU = cfg.n_up
        n_up_in = 1 + 2 + 4 + 2            # fix, belief, id_up, bias
        Wu = rng.normal(0, 1.0 / np.sqrt(nU), (nU, nU)) * 0.9
        self.W_up = nn.Parameter(torch.tensor(Wu, dtype=torch.float32))
        self.U_up = nn.Parameter(torch.tensor(rng.normal(0, 1.0, (nU, n_up_in)), dtype=torch.float32))
        up_mask = np.ones((nU, n_up_in), np.float32)
        if cfg.arch == "A":
            up_mask[:, 3:7] = 0.0           # no identity-at-location input: no identity x belief conjunction upstream
        up_mask[:, 7:9] = 0.0              # side bias never enters via learnable weights (it would be unlearned);
                                           # with bias_locus == "upstream" it is added to the upstream output instead
        self.register_buffer("up_mask", torch.tensor(up_mask))
        self.b_up = nn.Parameter(torch.zeros(nU))
        n_out = 3 if cfg.arch == "C" else 2
        self.W_o = nn.Parameter(torch.tensor(rng.normal(0, 0.1, (n_out, nU)), dtype=torch.float32))
        self.b_o = nn.Parameter(torch.tensor([-2.0] * n_out, dtype=torch.float32))

        # ---------------- readout + value baseline ----------------
        self.g_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(cfg.hazard_gain_init)))))
        self.theta = nn.Parameter(torch.tensor(cfg.hazard_theta_init))
        self.W_val = nn.Linear(N + nU + 1, 1)
        nn.init.zeros_(self.W_val.weight)
        nn.init.zeros_(self.W_val.bias)

        # ---------------- perturbation buffers ----------------
        self.register_buffer("w_gain", torch.ones(N, N), persistent=False)
        self.register_buffer("snr_gain", torch.ones(N), persistent=False)
        self.register_buffer("unit_gain", torch.ones(N), persistent=False)
        self.dis_override = None
        self.hand_upstream = None
        # fixed per-unit shape (orientation) preference; separate RNG stream so all other weights are unchanged
        u = np.random.default_rng(cfg.seed + 7919).normal(0, 1, N).astype(np.float32)
        self.register_buffer("shape_u", torch.tensor(u))

    # ------------------------------------------------------------------
    @property
    def W_sc(self):
        return torch.abs(self.W_raw) * self.sign * self.mask * self.w_gain

    def upstream(self, inp, noise=True, generator=None):
        """Return disinhibition [B, T, 2] and upstream rates [B, T, nU]."""
        cfg = self.cfg
        B, T, _ = inp["fix"].shape
        dev = inp["fix"].device
        if self.hand_upstream is not None:
            dis, R_up = self.hand_upstream(inp, noise, generator)
        else:
            u = torch.cat([inp["fix"], inp["belief"], inp["id_up"], inp["bias"]], -1)
            ext = u @ (self.U_up * self.up_mask).T + self.b_up
            x = torch.zeros(B, cfg.n_up, device=dev)
            sq = cfg.noise_up * np.sqrt(self.alpha_up)
            outs = []
            for i in range(T):
                r = F.softplus(x)
                x = (1 - self.alpha_up) * x + self.alpha_up * (r @ self.W_up.T + ext[:, i])
                if noise and cfg.noise_up > 0:
                    x = x + sq * torch.randn(x.shape, device=dev, generator=generator)
                outs.append(F.softplus(x))
            R_up = torch.stack(outs, 1)
            o = R_up @ self.W_o.T + self.b_o
            if cfg.bias_locus == "upstream":   # fixed-weight side bias on the per-location upstream outputs
                o = torch.cat([o[..., :2] + inp["bias"], o[..., 2:]], -1)
            if cfg.arch == "C":
                dis = cfg.c_dis_gain * cfg.dis_max * torch.softmax(cfg.c_beta * o, -1)[..., :2]
            else:
                dis = cfg.dis_max * torch.sigmoid(o)
        if cfg.dis_tau > 0:
            a = cfg.dt / cfg.dis_tau
            acc, ys = torch.zeros_like(dis[:, 0]), []
            for i in range(dis.shape[1]):
                acc = (1 - a) * acc + a * dis[:, i]
                ys.append(acc)
            dis = torch.stack(ys, 1)
        if self.dis_override is not None:
            dis = self.dis_override(dis, inp)
        return dis, R_up

    def forward(self, inp, noise=True, generator=None):
        """Returns dict: sc rates [B,T,N], snr [B,T,2], dis [B,T,2], up rates [B,T,nU], pool [B,T,2]."""
        cfg = self.cfg
        B, T, _ = inp["fix"].shape
        dev = inp["fix"].device
        dis, R_up = self.upstream(inp, noise, generator)
        snr = F.relu(cfg.snr_tonic + cfg.snr_fix * inp["fix"] - dis)                   # [B,T,2]
        # external input to SC units
        if cfg.shape_vis > 0:
            ext = 0
            for loc in (0, 1):
                fac = F.relu(1 + cfg.shape_vis * self.shape_u * inp["idsign"][:, None, loc:loc + 1])   # [B,1,N]
                ext = ext + inp["vis"][..., loc:loc + 1] * self.w_vis[:, loc] * fac
        else:
            ext = inp["vis"] @ self.w_vis.T
        ext = ext - cfg.w_snr * (snr @ self.pool_of.T) * self.snr_gain
        ext = ext + (inp["slow"] @ self.pool_of.T)
        if cfg.bias_locus == "SC":
            ext = ext + inp["bias"] @ self.pool_of.T
        if self.w_fix_sc is not None:
            ext = ext + inp["fix"] * self.w_fix_sc
        if self.w_a is not None:
            ext = ext + torch.cat([inp["id_sc"], inp["belief"]], -1) @ self.w_a.T
        ext = ext + self.b_sc
        W = self.W_sc
        g = self.unit_gain
        x = torch.zeros(B, self.idx.n, device=dev)
        sq = cfg.noise_sc * np.sqrt(self.alpha)
        rates = []
        for i in range(T):
            r = F.softplus(x) * g
            x = (1 - self.alpha) * x + self.alpha * (r @ W.T + ext[:, i])
            if noise and cfg.noise_sc > 0:
                x = x + sq * torch.randn(x.shape, device=dev, generator=generator)
            rates.append(F.softplus(x) * g)
        R = torch.stack(rates, 1)
        pool = torch.stack([R[..., self.idx.pool(0)].mean(-1), R[..., self.idx.pool(1)].mean(-1)], -1)
        return dict(sc=R, snr=snr, dis=dis, up=R_up, pool=pool)

    def hazard_logits(self, pool):
        return F.softplus(self.g_raw) * (pool - self.theta)

    def value(self, out, tfrac):
        return self.W_val(torch.cat([out["sc"], out["up"], tfrac], -1)).squeeze(-1)
