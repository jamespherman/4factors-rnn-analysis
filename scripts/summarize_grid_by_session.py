#!/usr/bin/env python3
"""
Per-session and session-group summaries of a context grid evaluated by
scripts/evaluate_context_models.py (reads <root>/evaluation_raw.npy).

For each variant and session group, per restart: held-out r between the model's per-neuron
effect and the recorded effect on test trials (pooled over the group's neurons), the
ceiling r(recorded non-test, recorded test), and the delta held-out LL against none_free
with the same restart. Reported as mean [min, max] over restarts.

Usage:
    python scripts/summarize_grid_by_session.py --root results/grid_v12
"""

import argparse
from pathlib import Path

import numpy as np

EFFECTS = ('reward_pre', 'block_pre', 'prob_vis', 'spatial')
BASELINE = 'none_free'


def heldout(runs: list, k: str, model_on_test: bool = False) -> tuple:
    """(model vs recorded-test r, ceiling) for effect k pooled over runs' neurons."""
    m = np.concatenate([r['effects_test' if model_on_test else 'effects'][k] for r in runs])
    rt = np.concatenate([r['recorded_split']['test'][k] for r in runs])
    rr = np.concatenate([r['recorded_split']['rest'][k] for r in runs])
    ok = ~np.isnan(m) & ~np.isnan(rt) & ~np.isnan(rr)
    if ok.sum() < 4:
        return np.nan, np.nan
    return np.corrcoef(m[ok], rt[ok])[0, 1], np.corrcoef(rr[ok], rt[ok])[0, 1]


def fmt(values) -> str:
    v = np.array([x for x in values if not np.isnan(x)])
    return '-' if len(v) == 0 else f'{v.mean():+.2f} [{v.min():+.2f}, {v.max():+.2f}]'


def main():
    parser = argparse.ArgumentParser(description='Per-session grid summary')
    parser.add_argument('--root', default='results/grid_v12')
    parser.add_argument('--model-on-test', action='store_true',
                        help='Model effects from the test trials only (same switches as the recorded test effect)')
    args = parser.parse_args()
    root = Path(args.root)
    results = np.load(root / 'evaluation_raw.npy', allow_pickle=True).item()

    variants = sorted({k.split('/')[0] for k in results})
    sessions = sorted({k.split('/')[1].rsplit('_seed', 1)[0] for k in results})
    seeds = sorted({k.rsplit('_seed', 1)[1] for k in results}, key=int)
    newton_effect = [s for s in sessions if s in ('08_13', '08_14')]
    groups = [('all sessions', sessions), ('Newton 08_13 + 08_14 (reward effect present)', newton_effect),
              ('other sessions', [s for s in sessions if s not in newton_effect])] + [(s, [s]) for s in sessions]

    lines = [f'# Context grid by session ({root})', '',
             'Per restart: r between model and recorded per-neuron effect on held-out test trials, pooled '
             'over the group\'s neurons; mean [min, max] over restarts. Ceiling: r(recorded non-test, '
             f'recorded test), mean [min, max] over splits. Delta LL: held-out Poisson LL per neuron-bin vs `{BASELINE}` (x1000).', '']
    for gname, gsess in groups:
        if not gsess:
            continue
        lines += [f'## {gname}', '',
                  '| Variant | Reward pre r | Block pre r | Prob vis r | Spatial r | Delta LL |',
                  '|---|---|---|---|---|---|']
        ceilings = {}
        for v in variants:
            per = {k: [] for k in EFFECTS}
            dll = []
            for sd in seeds:
                runs = [results.get(f'{v}/{s}_seed{sd}') for s in gsess]
                base = [results.get(f'{BASELINE}/{s}_seed{sd}') for s in gsess]
                if any(r is None for r in runs):
                    continue
                for k in EFFECTS:
                    r, c = heldout(runs, k, args.model_on_test)
                    per[k].append(r)
                    ceilings.setdefault(k, {})[sd] = c  # depends on the split only
                if all(b is not None for b in base):
                    dll.append(1000 * np.mean(np.concatenate([r['ll'] for r in runs])
                                              - np.concatenate([b['ll'] for b in base])))
            lines.append(f'| {v} | ' + ' | '.join(fmt(per[k]) for k in EFFECTS) + f' | {fmt(dll)} |')
        lines.append('| *ceiling* | ' + ' | '.join(fmt(list(ceilings.get(k, {}).values())) for k in EFFECTS) + ' | |')
        lines.append('')

    report = '\n'.join(lines) + '\n'
    name = 'by_session_report_model_on_test.md' if args.model_on_test else 'by_session_report.md'
    (root / name).write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
