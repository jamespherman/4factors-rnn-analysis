"""
Jitter-corrected cross-correlograms (CCGs) for detecting short-latency spike transmission.

Interval jitter (Amarasingham et al. 2012): post spikes are redistributed uniformly within fixed jitter
intervals, which keeps slower co-modulation (task, drift) and removes fine timing. The jitter-expected CCG is
computed analytically: each post spike contributes 1/J per ms of its J-ms interval. Spike transmission gain
(STG) = (observed - expected) / n_pre in the causal window; negative for inhibition.
"""

from typing import Dict

import numpy as np
from scipy.stats import poisson


def _grid(times: np.ndarray, t0: float, n_ms: int) -> np.ndarray:
    idx = np.floor((times - t0) * 1000).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_ms)]
    return np.bincount(idx, minlength=n_ms).astype(np.float32)


def ccg_jitter(pre: np.ndarray, post: np.ndarray, t0: float, t1: float, max_lag: int = 50, jitter_ms: int = 25,
               roi=(1, 4), flank=(-4, -1)) -> Dict[str, np.ndarray]:
    """
    CCG of post relative to pre (1-ms bins, lags -max_lag..max_lag ms) within [t0, t1] s, with interval-jitter
    expectation and tests on the causal window roi (ms, inclusive) and the anticausal flank.

    Returns dict with lags, observed, expected, n_pre, stg, p_excess, p_deficit (Poisson on the roi sum),
    flank_stg, min_detectable_stg (z = 3.3) and coincident fraction within +-0.5 ms.
    """
    n_ms = int(np.floor((t1 - t0) * 1000))
    n_ms -= n_ms % jitter_ms
    post_g = _grid(post, t0, n_ms)
    smeared = np.repeat(post_g.reshape(-1, jitter_ms).sum(1) / jitter_ms, jitter_ms)
    pre_idx = np.floor((pre[(pre >= t0) & (pre < t1)] - t0) * 1000).astype(np.int64)
    pre_idx = pre_idx[(pre_idx >= max_lag) & (pre_idx < n_ms - max_lag)]
    lags = np.arange(-max_lag, max_lag + 1)
    obs = np.array([post_g[pre_idx + k].sum() for k in lags])
    exp = np.array([smeared[pre_idx + k].sum() for k in lags])
    sel = (lags >= roi[0]) & (lags <= roi[1])
    fl = (lags >= flank[0]) & (lags <= flank[1])
    o, e = obs[sel].sum(), exp[sel].sum()
    n_pre = len(pre_idx)
    zero = lags == 0
    return {
        'lags': lags, 'observed': obs, 'expected': exp, 'n_pre': n_pre, 'n_post': int(post_g.sum()),
        'stg': (o - e) / max(n_pre, 1), 'flank_stg': (obs[fl].sum() - exp[fl].sum()) / max(n_pre, 1),
        'p_excess': float(poisson.sf(o - 1, e)) if e > 0 else 1.0,
        'p_deficit': float(poisson.cdf(o, e)) if e > 0 else 1.0,
        'min_detectable_stg': 3.3 * np.sqrt(e) / max(n_pre, 1),
        'coincident_frac': float(obs[zero].sum() / max(min(n_pre, post_g.sum()), 1)),
    }
