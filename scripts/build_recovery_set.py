#!/usr/bin/env python3
"""
Teachers, synthetic sessions and the student job list for the recovery test (Stage D) on
every session of a context grid.

Per session, teachers are the grid restarts with the best selection-set likelihood:
    null    none_free   (no context)
    ext     free_free   (per-unit context into all units), g = 1
    local   freeI_free  (per-unit context into I units only), g = 1
    shared  shared_free (one shared context signal), g chosen so that the SD across neurons
            of its context-driven reward effect matches the ext teacher's (the pilot's g = 1
            shared teacher had almost no context effect)

Students: none / shared / sharedI / freeI / free, each with --restarts restarts, fit exactly
as on real data.

With --manifest, teachers and gains are reused from an earlier set; with --drift-amp, the
synthetic sessions also carry the recorded neurons' slow fluctuations, shifted out of
alignment with context (make_synthetic_data.py; calibrate with calibrate_teacher_drift.py).

Usage:
    python scripts/build_recovery_set.py --grid results/grid_v12 --restarts 3
    python scripts/build_recovery_set.py --manifest results/recovery_v12/manifest.json \
        --drift-amp null=2,ext=1.25,local=1.5,shared=2 --restarts 6 \
        --data-dir data/synthetic_v12_drift --results-dir results/recovery_v12_drift \
        --jobs slurm/jobs_recovery_v12_drift.txt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.sessions import ALL_SESSIONS, load_export
from evaluate_context_models import load_context_model
from teacher_effect_size import model_effects
import analyze_context_effects as A

TEACHERS = {'null': 'none_free', 'ext': 'free_free', 'local': 'freeI_free', 'shared': 'shared_free'}
STUDENTS = {'none_free': '--context none', 'shared_free': '--context shared',
            'sharedI_free': '--context shared --context-targets i',
            'freeI_free': '--context free --context-targets i', 'free_free': '--context free'}
SHARED_GAINS = [1, 2, 3, 5, 8]


def best_restart(grid: Path, variant: str, session: str) -> Path:
    dirs = sorted(grid.glob(f'{variant}/{session}_seed*'))
    assert dirs, f"No {variant} runs for {session}"
    return max(dirs, key=lambda d: json.loads((d / 'summary.json').read_text())['select_ll'])


def context_sd(model_dir: Path, gain: float, base: np.ndarray, sess: dict, data: dict) -> float:
    """SD across neurons of the context-driven part of the reward-context effect."""
    return float(np.nanstd(model_effects(model_dir, gain, sess, data)['reward_pre'] - base))


def main():
    parser = argparse.ArgumentParser(description='Build the recovery-test set')
    parser.add_argument('--grid', default='results/grid_v12')
    parser.add_argument('--sessions', default='all')
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--manifest', default=None, help='Reuse teachers and gains from this manifest')
    parser.add_argument('--drift-amp', default='0',
                        help="Slow-fluctuation strength: one value, or per teacher as 'null=2,ext=1.25,...'")
    parser.add_argument('--data-dir', default='data/synthetic_v12')
    parser.add_argument('--results-dir', default='results/recovery_v12')
    parser.add_argument('--jobs', default='slurm/jobs_recovery_v12.txt')
    args = parser.parse_args()
    grid, data_dir = Path(args.grid), Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sessions = ALL_SESSIONS if args.sessions == 'all' else args.sessions.split(',')

    amps = ({k: float(v) for k, v in (kv.split('=') for kv in args.drift_amp.split(','))}
            if '=' in args.drift_amp else {t: float(args.drift_amp) for t in TEACHERS})
    lines, manifest = [], {}
    prior = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
    for session in sessions:
        if session in prior:
            teachers = {t: Path(d) for t, d in prior[session]['teachers'].items()}
            gains = prior[session]['gains']
            manifest[session] = dict(prior[session], drift_amp=amps)
        else:
            teachers = {t: best_restart(grid, v, session) for t, v in TEACHERS.items()}
            _, config = load_context_model(teachers['ext'])
            data = load_export(config['session'], config['trials'])
            sess = A.session_from_data(data, session)
            gains = {'null': 1.0, 'ext': 1.0, 'local': 1.0}
            ref_base = model_effects(teachers['ext'], 0.0, sess, data)['reward_pre']
            ref_sd = context_sd(teachers['ext'], 1.0, ref_base, sess, data)
            sh_base = model_effects(teachers['shared'], 0.0, sess, data)['reward_pre']
            sds = {g: context_sd(teachers['shared'], g, sh_base, sess, data) for g in SHARED_GAINS}
            gains['shared'] = float(min(SHARED_GAINS, key=lambda g: abs(np.log(max(sds[g], 1e-6) / ref_sd))))
            manifest[session] = {'teachers': {t: d.as_posix() for t, d in teachers.items()}, 'gains': gains,
                                 'ext_context_sd': ref_sd, 'shared_context_sd_by_gain': sds,
                                 'drift_amp': amps}
            print(f"{session}: ext context SD {ref_sd:.2f}; shared "
                  f"{', '.join(f'g{g}={v:.2f}' for g, v in sds.items())} -> g = {gains['shared']}", flush=True)

        for t, d in teachers.items():
            name = f'{t}_{session}'
            out = data_dir / f'{name}.mat'
            if not out.exists():
                subprocess.run([sys.executable, 'scripts/make_synthetic_data.py', '--teacher', str(d),
                                '--ctx-scale', str(gains[t]), '--drift-amp', str(amps[t]), '--out', str(out)], check=True)
            for seed in range(1, args.restarts + 1):
                for student, flags in STUDENTS.items():
                    lines.append(f"--session {session} --data {out.as_posix()} --trials all --lwin-scale 1 "
                                 f"--lctx-scale 0 --split switch {flags} --ie-mode free --seed {seed} "
                                 f"--output-dir {args.results_dir}/{name}/{student}/{session}_seed{seed}")

    with open(args.jobs, 'w', newline='\n') as f:  # LF endings: read by bash on CRC
        f.write('\n'.join(lines) + '\n')
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"{len(lines)} student runs -> {args.jobs}")


if __name__ == '__main__':
    main()
