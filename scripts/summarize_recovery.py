#!/usr/bin/env python3
"""
Summary of the recovery test (Stage D) built by scripts/build_recovery_set.py, after each
dataset <root>/<teacher>_<session> was evaluated by scripts/evaluate_context_models.py.

For each teacher type and student variant, per restart: held-out r between the student's
per-neuron effect and the synthetic "recorded" (teacher-generated) effect on test trials,
pooled over the sessions' neurons; the ceiling r(recorded non-test, recorded test); and the
delta held-out LL against the none_free student with the same restart. Reported as
mean [min, max] over restarts, pooled over all sessions and per session (reward pre only).

Usage:
    python scripts/summarize_recovery.py --root results/recovery_v12 --model-on-test
"""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_grid_by_session import heldout, fmt, EFFECTS, BASELINE

TEACHERS = ('null', 'ext', 'local', 'shared')
STUDENTS = ('none_free', 'shared_free', 'sharedI_free', 'freeI_free', 'free_free')


def main():
    parser = argparse.ArgumentParser(description='Recovery-test summary')
    parser.add_argument('--root', default='results/recovery_v12')
    parser.add_argument('--model-on-test', action='store_true',
                        help='Student effects from the test trials only (same switches as the recorded test effect)')
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root / 'manifest.json').read_text())
    sessions = list(manifest)

    # results[teacher][session] = {'<student>/<session>_seed<k>': evaluation}
    results = {t: {s: np.load(root / f'{t}_{s}' / 'evaluation_raw.npy', allow_pickle=True).item()
                   for s in sessions} for t in TEACHERS}
    seeds = sorted({k.rsplit('_seed', 1)[1] for r in results['null'].values() for k in r}, key=int)

    def runs_for(t, student, sd, sess_list):
        runs = [results[t][s].get(f'{student}/{s}_seed{sd}') for s in sess_list]
        return None if any(r is None for r in runs) else runs

    gains = ', '.join(f"{s} {manifest[s]['gains']['shared']:g}" for s in sessions)
    lines = [f'# Recovery test ({root})', '',
             'Teachers per session: best grid restart by selection LL; null = none_free, ext = free_free, '
             'local = freeI_free (g = 1), shared = shared_free with gain matched to ext\'s context-driven '
             f'effect SD ({gains}). Students fit as on real data (switch split, restart = split seed).', '',
             'r: student vs teacher-generated per-neuron effect on held-out test trials'
             + (' (student effect also from test trials)' if args.model_on_test else '')
             + ', pooled over sessions; mean [min, max] over restarts. Delta LL vs none_free student (x1000).', '']
    summary = {}
    for t in TEACHERS:
        lines += [f'## Teacher: {t}', '',
                  '| Student | Reward pre r | Block pre r | Prob vis r | Spatial r | Delta LL |',
                  '|---|---|---|---|---|---|']
        ceilings = {}
        for st in STUDENTS:
            per, dll = {k: [] for k in EFFECTS}, []
            for sd in seeds:
                runs, base = runs_for(t, st, sd, sessions), runs_for(t, BASELINE, sd, sessions)
                if runs is None:
                    continue
                for k in EFFECTS:
                    r, c = heldout(runs, k, args.model_on_test)
                    per[k].append(r)
                    ceilings.setdefault(k, {})[sd] = c
                if base is not None:
                    dll.append(1000 * np.mean(np.concatenate([r['ll'] for r in runs])
                                              - np.concatenate([b['ll'] for b in base])))
            summary.setdefault(t, {})[st] = {'reward_pre_r': per['reward_pre'], 'dll': dll}
            lines.append(f'| {st} | ' + ' | '.join(fmt(per[k]) for k in EFFECTS) + f' | {fmt(dll)} |')
        lines.append('| *ceiling* | ' + ' | '.join(fmt(list(ceilings.get(k, {}).values())) for k in EFFECTS) + ' | |')
        lines.append('')

    lines += ['## Reward pre r per session', '', '| Teacher | Session | ' + ' | '.join(STUDENTS) + ' | ceiling |',
              '|---|---|' + '---|' * (len(STUDENTS) + 1)]
    for t in TEACHERS:
        for s in sessions:
            cells, ceil = [], []
            for st in STUDENTS:
                vals = []
                for sd in seeds:
                    runs = runs_for(t, st, sd, [s])
                    if runs is not None:
                        r, c = heldout(runs, 'reward_pre', args.model_on_test)
                        vals.append(r)
                        ceil.append(c)
                cells.append(fmt(vals).split(' [')[0])
            lines.append(f'| {t} | {s} | ' + ' | '.join(cells) + f' | {fmt(ceil).split(" [")[0]} |')

    report = '\n'.join(lines) + '\n'
    name = 'recovery_report_model_on_test.md' if args.model_on_test else 'recovery_report.md'
    (root / name).write_text(report, encoding='utf-8')
    (root / name.replace('.md', '.json')).write_text(json.dumps(summary, indent=1, default=float))
    print(report)


if __name__ == '__main__':
    main()
