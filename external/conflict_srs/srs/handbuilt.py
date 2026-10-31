"""
Hand-built (untrained) B, C and A networks for Stage 0 (plan v2), using the same SRSNet class so that simulate()
and all perturbation hooks apply unchanged.

SC (all three): structured pools.
  E_pool(i) -> E_pool(i)                 self-excitation  w_ee (total over the pool)
  E_pool(i) -> I_local(same SC) -> E_pool(i)   local feedback inhibition (w_ei, w_ie)
  E_pool(i) -> I_comm(same SC)  -> E_pool(other SC)   cross-SC WTA (w_ec, w_ce), scaled by `wta`
  pool bias b_pool; other E / I units quiet (bias -2) unless used.
Upstream:
  B  v_i(t) = lowpass_tau_up( sum_k id_up[k@i] * V_k / value_scale ) + noise; dis_i = value_gain * v_i (graded)
  C  same v_i + decision noise, then dis = c_gain * softmax(beta_c * [v_L, v_R, nogo])[:2] (near-categorical;
     c_gain defaults to cfg.dis_max; lower it to slow C's RT);
     SC WTA weak (wta_C)
  A  upstream delivers no value (dis = 0); SC E_other units act as conjunction units: for location f and identity k,
     a group of E_other units in the SC representing f receives beta_conj * (id_k@f + V_k/value_scale - 1) and
     projects to pool f (w_conj). Binding identity x belief x location therefore happens inside SC.
All parameters are in HB_DEFAULTS and can be overridden (e.g. by a Stage 0 calibration).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .config import SRSConfig
from .network import SRSNet

HB_DEFAULTS = dict(
    w_ee=1.2, w_ei=1.0, w_ie=1.0, w_ec=1.0, w_ce=2.0, wta=1.0, wta_C=0.2,
    b_pool=-0.5, b_quiet=-2.0, theta=0.75, gain=20.0,
    value_gain=1.5, up_noise=0.0, beta_c=30.0, nogo=0.15, c_gain=None,
    beta_conj=6.0, w_conj=0.25, fix_sc=0.0,
)


def _lowpass(x, alpha):
    """x [B,T,C] -> first-order low-pass along T."""
    y = torch.zeros_like(x)
    acc = torch.zeros_like(x[:, 0])
    for i in range(x.shape[1]):
        acc = (1 - alpha) * acc + alpha * x[:, i]
        y[:, i] = acc
    return y


def build_handbuilt(arch: str, cfg: SRSConfig = None, **params):
    cfg = cfg or SRSConfig()
    cfg.arch = arch
    p = {**HB_DEFAULTS, **params}
    net = SRSNet(cfg)
    idx = net.idx
    N = idx.n
    W = np.zeros((N, N), np.float32)
    wta = p["wta_C"] if arch == "C" else p["wta"]
    for s in (0, 1):
        pool = np.where((idx.side == s) & (idx.type == 0))[0]
        il = np.where((idx.side == s) & (idx.type == 2))[0]
        ic = np.where((idx.side == s) & (idx.type == 3))[0]
        other_pool = np.where((idx.side == 1 - s) & (idx.type == 0))[0]
        W[np.ix_(pool, pool)] = p["w_ee"] / len(pool)
        W[np.ix_(il, pool)] = p["w_ei"] / len(pool)
        W[np.ix_(pool, il)] = p["w_ie"] / len(il)              # magnitude; sign applied by Dale
        W[np.ix_(ic, pool)] = p["w_ec"] * wta / len(pool)
        W[np.ix_(other_pool, ic)] = p["w_ce"] * wta / len(ic)
    np.fill_diagonal(W, 0)
    with torch.no_grad():
        net.W_raw.copy_(torch.tensor(W))
        net.mask.copy_(torch.tensor((W != 0).astype(np.float32)))
        b = np.full(N, p["b_quiet"], np.float32)
        b[idx.is_pool] = p["b_pool"]
        if net.w_fix_sc is not None:
            net.w_fix_sc.fill_(p["fix_sc"])
        net.g_raw.fill_(float(np.log(np.expm1(p["gain"]))))
        net.theta.fill_(p["theta"])
        if arch == "A":
            # conjunction units: in each SC, E_other split into 2 groups (identity 1, 2) for that SC's field
            Wa = np.zeros((N, 6), np.float32)            # columns: T1@L, T2@L, T1@R, T2@R, V1, V2
            for s in (0, 1):
                f = 1 - s
                eo = np.where((idx.side == s) & (idx.type == 1))[0]
                pool = np.where((idx.side == s) & (idx.type == 0))[0]
                halves = np.array_split(eo, 2)
                for k, grp in zip((1, 2), halves):
                    Wa[grp, 2 * f + (k - 1)] = p["beta_conj"]
                    Wa[grp, 4 + (k - 1)] = p["beta_conj"] * 1.0
                    b[grp] = -p["beta_conj"]
                    W[np.ix_(pool, grp)] = p["w_conj"] / len(grp) * 1.0
            net.W_raw.copy_(torch.tensor(W))
            net.mask.copy_(torch.tensor((W != 0).astype(np.float32)))
            net.w_a = torch.tensor(Wa, device=net.W_raw.device)
        net.b_sc.copy_(torch.tensor(b))

    alpha_up = cfg.dt / cfg.tau_up

    def hand_upstream(inp, noise, generator):
        B, T, _ = inp["fix"].shape
        dev = inp["fix"].device
        R_up = torch.zeros(B, T, cfg.n_up, device=dev)
        if arch == "A":
            return torch.zeros(B, T, 2, device=dev), R_up
        idu = inp["id_up"]                                   # [B,T,4]: T1@L, T2@L, T1@R, T2@R
        V = inp["belief"]                                    # [B,T,2] (already / value_scale)
        v = torch.stack([idu[..., 0] * V[..., 0] + idu[..., 1] * V[..., 1],
                         idu[..., 2] * V[..., 0] + idu[..., 3] * V[..., 1]], -1)
        v = _lowpass(v, alpha_up)
        if noise and p["up_noise"] > 0:
            v = v + p["up_noise"] * _lowpass(torch.randn(v.shape, device=dev, generator=generator), alpha_up) \
                * np.sqrt(2 / alpha_up)
        if arch == "B":
            dis = p["value_gain"] * v.clamp(min=0)
        else:
            nogo = torch.full_like(v[..., :1], p["nogo"])
            gain_c = cfg.dis_max if p["c_gain"] is None else p["c_gain"]
            dis = gain_c * torch.softmax(p["beta_c"] * torch.cat([v, nogo], -1), -1)[..., :2]
        return dis, R_up

    net.hand_upstream = hand_upstream
    net.handbuilt_params = p
    return net
