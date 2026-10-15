#!/usr/bin/env python3
"""
Calibrate the slow-fluctuation strength of recovery teachers (make_synthetic_data.py
--drift-amp) so that the reliability of the synthetic reward-context pattern across reward
switches matches the recordings.

For each teacher type and drift amplitude a, synthetic sessions are generated for every
session in the recovery manifest, and the switch-split ceiling r(effect on held-out switch
windows, effect on the rest) is computed exactly as in evaluate_context_models.py, pooled
over sessions, for split seeds 1..n. The recorded ceiling is computed the same way.

Usage:
    python scripts/calibrate_teacher_drift.py --manifest results/recovery_v12/manifest.json --amps 0,1,2,3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.sessions import mask_unstable
from src.analysis.task_context import get_task_context
from make_synthetic_data import synthesize
from train_context import switch_window_split, SWITCH_HALF_WIDTH
from evaluate_context_models import effects
import analyze_context_effects as A

KEYS = ('reward_pre', 'block_pre')


def split_effects(data: dict, session: str, seeds: list) -> dict:
    """{seed: {'test': effects, 'rest': effects}} for the switch split with that seed."""
    sess = A.session_from_data(data, session)
    out = {}
    for sd in seeds:
        _, _, test = switch_window_split(get_task_context(data)['reward_map'], 2, 2, SWITCH_HALF_WIDTH, sd)
        is_test = np.isin(np.arange(sess['pre'].shape[0]), test)
        out[sd] = {}
        for name, keep in [('test', is_test), ('rest', ~is_test)]:
            part = dict(sess)
            part['vis'] = np.where(keep[:, None], sess['vis'], np.nan)
            part['pre'] = np.where(keep[:, None], sess['pre'], np.nan)
            out[sd][name] = effects(part, min_trials=2)
    return out


def pooled_ceiling(per_session: list, seeds: list, k: str) -> list:
    vals = []
    for sd in seeds:
        rt = np.concatenate([p[sd]['test'][k] for p in per_session])
        rr = np.concatenate([p[sd]['rest'][k] for p in per_session])
        ok = ~np.isnan(rt) & ~np.isnan(rr)
        vals.append(np.corrcoef(rt[ok], rr[ok])[0, 1])
    return vals


def main():
    parser = argparse.ArgumentParser(description='Calibrate teacher slow-fluctuation strength')
    parser.add_argument('--manifest', default='results/recovery_v12/manifest.json')
    parser.add_argument('--amps', default='0,1,2,3')
    parser.add_argument('--teachers', default='null,ext,local,shared')
    parser.add_argument('--n-seeds', type=int, default=6)
    parser.add_argument('--drift-sigma', type=float, default=15.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    seeds = list(range(1, args.n_seeds + 1))
    amps = [float(a) for a in args.amps.split(',')]

    rows, recorded = {}, []
    for t in args.teachers.split(','):
        for a in amps:
            per = []
            for s, m in manifest.items():
                synthetic, data, _ = synthesize(Path(m['teachers'][t]), m['gains'][t], drift_amp=a, drift_sigma=args.drift_sigma)
                per.append(split_effects(mask_unstable(synthetic), s, seeds))
                if t == args.teachers.split(',')[0] and a == amps[0]:
                    recorded.append(split_effects(data, s, seeds))
            rows[(t, a)] = {k: pooled_ceiling(per, seeds, k) for k in KEYS}
            print(f"{t:7s} a={a:g}: " + '  '.join(f"{k} {np.mean(v):+.2f} [{min(v):+.2f}, {max(v):+.2f}]"
                                                    for k, v in rows[(t, a)].items()), flush=True)
    rec = {k: pooled_ceiling(recorded, seeds, k) for k in KEYS}
    print('recorded:  ' + '  '.join(f"{k} {np.mean(v):+.2f} [{min(v):+.2f}, {max(v):+.2f}]" for k, v in rec.items()))
    if args.out:
        Path(args.out).write_text(json.dumps({'recorded': rec, 'synthetic': {f'{t}_a{a:g}': v for (t, a), v in rows.items()}},
                                             indent=1))


if __name__ == '__main__':
    main()
