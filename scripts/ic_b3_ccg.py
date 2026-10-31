#!/usr/bin/env python3
"""
Phase 3 / B3 (plans/2026-10-31_intercollicular_plan.md): within-SC jitter-corrected CCGs.

All ordered pairs of exported units on different channels; spikes from the span of trials stable for both
(first stable trial's fixation onset - 1 s to last stable trial's saccade offset + 1 s, ITIs included).
1-ms bins, 25-ms interval jitter (src/ccg.py, validated on synthetic trains). Causal window 1-4 ms.
Significant: Poisson p < 0.001 on the window sum and |causal STG| > |anticausal (-4..-1 ms) STG|.
Probable duplicates (zero-lag coincidence > 0.1) excluded. Per-pair minimum detectable STG (z = 3.3).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.spikes import load_spikes  # noqa: E402
from src.ccg import ccg_jitter  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'b3_ccg'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ALL_SESSIONS:
        path = OUT / f'{s}.npy'
        if path.exists():
            continue
        d = load_spikes(s)
        ids, st = d['ids'], d['stable']
        rows = []
        for a in range(len(ids)):
            for b in range(len(ids)):
                if a == b or d['info_ch'][a] == d['info_ch'][b]:
                    continue
                both = np.flatnonzero(st[a] & st[b])
                if len(both) < 100:
                    continue
                t0 = np.nanmin(d['ev_fixOn'][both]) - 1
                t1 = np.nanmax(d['ev_saccadeOffset'][both]) + 1
                r = ccg_jitter(d[f'spk_{ids[a]}'], d[f'spk_{ids[b]}'], t0, t1)
                rows.append({'session': s, 'pre': int(ids[a]), 'post': int(ids[b]),
                             'pre_type': 'E' if d['neuron_type'][a] == 1 else 'I',
                             'post_type': 'E' if d['neuron_type'][b] == 1 else 'I',
                             'dist': abs(d['info_depth'][a] - d['info_depth'][b]), 'duration': t1 - t0,
                             **{k: r[k] for k in ('n_pre', 'n_post', 'stg', 'flank_stg', 'p_excess', 'p_deficit',
                                                  'min_detectable_stg', 'coincident_frac')},
                             'ccg_obs': r['observed'].astype(np.int32), 'ccg_exp': r['expected'].astype(np.float32)})
        np.save(path, rows, allow_pickle=True)
        print(s, len(rows), 'pairs', flush=True)


if __name__ == '__main__':
    main()
