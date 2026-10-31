#!/usr/bin/env python3
"""
Spatial response profile of classic SC neurons and putative interneurons, and which classic criterion each
putative interneuron fails.

Reads the pipeline's cached screening tests (session_data.cached_friedman_results in the processed
session_data.mat files: per location x epoch paired t-tests of epoch vs baseline, Bonferroni-corrected, and
mean rates per location and epoch) and reconstructs the classic-SC rule of screen_sc_neurons.m (2026-10-28):
    (a) significant activation at >= 1 contralateral location (visual, delay or saccade epoch)
    (c) every significant ipsilateral activation reliably < half of contralateral (Welch test; not
        recomputed here: per-trial location groups aren't cached; a neuron is only flagged)
    (d) no significant suppression at any task location in the visual or saccade epoch
Locations: the four gSac_4factors locations; the cache's fifth group (gSac_jph) counts for (a) only.

Also joins the visual-epoch location ROC from compare_selectivity_by_type.py.

Usage:
    python scripts/sc_spatial_profile.py
"""

from pathlib import Path
import itertools
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS, load_export, session_name
from src.analysis.pipeline_selectivity import get_target_onset_bins, compute_window_rates

PIPELINE = Path(r'D:/OneDrive - University of Pittsburgh/Code/4factors-analysis-pipeline/data/processed')
OUT = Path(__file__).parent.parent / 'results' / 'sc_spatial_profile'
EPOCHS = ['visual', 'delay', 'saccade']


def chars(d) -> str:
    return ''.join(map(chr, np.asarray(d).ravel()))


def load_cache(name: str) -> dict:
    """Cached screening tests for one session, indexed [cluster, location group, epoch]."""
    with h5py.File(PIPELINE / name / f'{name}_session_data.mat', 'r') as f:
        c = f['session_data/cached_friedman_results']
        si = f['session_data/analysis/screening_info']
        return {
            'cluster_ids': np.asarray(c['cluster_ids']).ravel().astype(int),
            'angles': np.asarray(c['location_angles']).ravel(),
            'sig': np.asarray(c['sig_by_location']).transpose(2, 1, 0).astype(bool),
            'supp': np.asarray(c['suppression_by_location']).transpose(2, 1, 0).astype(bool),
            # [groups, clusters, (baseline, visual, delay, saccade)]
            'means': np.stack([np.asarray(f[r]).T for r in np.asarray(c['group_mean_frs']).ravel()]),
            'side': chars(f['session_data/analysis/scSide']),
            'method': chars(c['method']), 'date': chars(c['computation_date']),
            'class': np.array([chars(f[r]) for r in np.asarray(si['neuron_class']).ravel()]),
            'class_ids': np.array([int(np.asarray(f[r]).ravel()[0]) for r in np.asarray(si['cluster_id']).ravel()]),
        }


def main():
    roc = np.load(Path(__file__).parent.parent / 'results' / 'selectivity_by_type' / 'grid_v13_free_free.npz')
    rows = []
    for s in ALL_SESSIONS:
        name = session_name(s)
        cache = load_cache(name)
        data = load_export(s, 'verified')
        ids = np.asarray(data['neuron_ids']).ravel().astype(int)
        etype = np.asarray(data['neuron_type']).ravel()
        col = np.array([np.flatnonzero(cache['cluster_ids'] == i)[0] for i in ids])
        pclass = np.array([cache['class'][cache['class_ids'] == i][0] for i in ids])
        assert np.all((pclass == 'classic') == (etype == 1)), f'{s}: export type differs from pipeline class'
        angles = cache['angles']
        contra = (angles > 90) & (angles < 270) if cache['side'] == 'right' else (angles < 90) | (angles > 270)
        four = np.arange(4)
        contra_x = np.cos(np.deg2rad(angles[:4])) * (-1 if cache['side'] == 'right' else 1)
        vertical = np.abs(contra_x) < 0.05

        # Consistency check: some assignment of cache groups 1-4 to export trial_location 1-4 (it varies by
        # session) makes the visual-epoch means agree; contra/ipsi below come from the cache's own angles
        win = compute_window_rates(np.transpose(data['firing_rates'], (2, 1, 0)), get_target_onset_bins(data))
        loc = np.asarray(data['trial_location']).ravel()
        exp_means = np.stack([np.nanmean(win[loc == k + 1], 0) for k in four])          # [4, neurons]
        exp_means -= exp_means.mean(0)
        r, perm = max((np.corrcoef(exp_means.ravel(), (cache['means'][list(q)][:, col, 1]
                                                        - cache['means'][list(q)][:, col, 1].mean(0)).ravel())[0, 1], q)
                      for q in itertools.permutations(range(4)))
        assert r > 0.9, f'{s}: no location assignment matches the export (r={r:.2f})'

        sel = roc['session'] == s
        assert sel.sum() == len(ids)
        for j, c in enumerate(col):
            sig, supp = cache['sig'][c], cache['supp'][c]
            evoked = cache['means'][:4, c, 1:] - cache['means'][:4, c, :1]   # [4 locations, 3 epochs]
            ci = []
            for e in range(3):
                bc, bi = evoked[contra[:4], e].max(), evoked[~contra[:4], e].max()
                ci.append((bc - bi) / (abs(bc) + abs(bi)) if abs(bc) + abs(bi) > 0 else np.nan)
            rows.append({
                'session': s, 'monkey': 'Feynman' if s.startswith('Feynman') else 'Newton', 'id': ids[j],
                'type': 'E' if etype[j] == 1 else 'I',
                'contra_act': bool(sig[contra].any()),
                'ipsi_act': bool(sig[four][~contra[:4]].any()),
                'supp_vis_sac': bool(supp[four][:, [0, 2]].any()),
                'n_act': sig[four].any(1).sum(),
                'act_contra_locs': sig[four][contra[:4]].any(1).sum(), 'act_ipsi_locs': sig[four][~contra[:4]].any(1).sum(),
                'act_epochs': sig[four].any(0), 'ci': np.array(ci), 'evoked': evoked,
                'contra_cols': contra[:4], 'angles': angles[:4], 'contra_x': contra_x,
                'ipsi_act_off_meridian': bool(sig[four][~contra[:4] & ~vertical].any()),
                'any_vertical': bool(vertical.any()),
                'roc_loc': roc['recorded_location'][sel][j, 0] / 2 + 0.5, 'roc_p': roc['recorded_location'][sel][j, 1],
            })
        print(f'{s}: angles {angles}, side {cache["side"]}, contra {angles[:4][contra[:4]]}, cache {cache["method"]} {cache["date"]}, '
              f'export locations 1-4 = cache groups {[q + 1 for q in perm]} (r={r:.2f})')
    report(rows)
    figure(rows)


def pattern(r) -> str:
    if r['act_contra_locs'] and r['act_ipsi_locs']:
        return 'both hemifields'
    if r['act_contra_locs']:
        return 'contra only'
    if r['act_ipsi_locs']:
        return 'ipsi only'
    return 'no activation (suppression only)'


def reasons(r) -> str:
    out = []
    if not r['contra_act']:
        out.append('(a) no contra activation')
    if r['ipsi_act']:
        out.append('(c) ipsi activation')
    if r['supp_vis_sac']:
        out.append('(d) vis/sac suppression')
    return ' + '.join(out) if out else 'none flagged'


def report(rows):
    lines = []
    p = lines.append
    for m in ['Feynman', 'Newton']:
        for t, label in [('I', 'putative interneurons'), ('E', 'classic SC')]:
            g = [r for r in rows if r['monkey'] == m and r['type'] == t]
            p(f'\n## {m}, {label} (n={len(g)})')
            if t == 'I':
                p('Failed criteria:')
                for k, n in sorted({x: sum(reasons(r) == x for r in g) for x in {reasons(r) for r in g}}.items(),
                                   key=lambda kv: -kv[1]):
                    p(f'  {k:55s} {n:3d} ({n / len(g):.0%})')
            p('Activation pattern over the 4 task locations (any epoch):')
            for k in ['contra only', 'both hemifields', 'ipsi only', 'no activation (suppression only)']:
                n = sum(pattern(r) == k for r in g)
                if n:
                    sub = [r for r in g if pattern(r) == k]
                    p(f'  {k:35s} {n:3d} ({n / len(g):.0%})  median visual location ROC '
                      f'{np.nanmedian([r["roc_loc"] for r in sub]):.2f}')
            act = np.array([r['act_epochs'] for r in g])
            p('Activated (any location) in epoch: ' + ', '.join(f'{e} {act[:, i].mean():.0%}' for i, e in enumerate(EPOCHS)))
            ci = np.array([r['ci'] for r in g])
            p('Contra-ipsi index of evoked response, median: ' +
              ', '.join(f'{e} {np.nanmedian(ci[:, i]):+.2f}' for i, e in enumerate(EPOCHS)))
            vm = [r for r in g if r['any_vertical']]
            if vm:
                p(f'Sessions with vertical-meridian targets (counted as ipsilateral): {len(vm)} neurons; ipsi '
                  f'activation flagged {sum(r["ipsi_act"] for r in vm)}, of which only at the vertical meridian '
                  f'{sum(r["ipsi_act"] and not r["ipsi_act_off_meridian"] for r in vm)}'
                  + (f'; non-classic only because of that: '
                     f'{sum(r["ipsi_act"] and not r["ipsi_act_off_meridian"] and r["contra_act"] and not r["supp_vis_sac"] for r in vm)}'
                     if t == 'I' else ''))
            if t == 'E':
                weak = [r for r in g if r['roc_loc'] < 0.6]
                p(f'Visual location ROC < 0.6: {len(weak)}; of these activated in visual epoch '
                  f'{sum(r["act_epochs"][0] for r in weak)}, delay {sum(r["act_epochs"][1] for r in weak)}, '
                  f'saccade {sum(r["act_epochs"][2] for r in weak)}; saccade-epoch contra-ipsi index median '
                  f'{np.nanmedian([r["ci"][2] for r in weak]):+.2f}')
    text = '\n'.join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.txt').write_text(text.lstrip() + '\n', encoding='utf-8')


def figure(rows):
    """Evoked response (epoch - baseline) at the 4 locations, contra first, per neuron, by monkey and class."""
    order = ['contra only', 'both hemifields', 'ipsi only', 'no activation (suppression only)']
    fig, axes = plt.subplots(1, 4, figsize=(16, 9.5))
    for ax, (m, t) in zip(axes, [('Feynman', 'E'), ('Feynman', 'I'), ('Newton', 'E'), ('Newton', 'I')]):
        g = [r for r in rows if r['monkey'] == m and r['type'] == t]
        g.sort(key=lambda r: (order.index(pattern(r)), -r['roc_loc']))
        mats = []
        for r in g:
            # columns per epoch: contra (higher angle sin first), then ipsi
            idx = np.argsort(-r['contra_x'], kind='stable')   # most contralateral first
            ev = r['evoked'][idx]                               # [4 locations, 3 epochs]
            mats.append((ev / max(np.abs(ev).max(), 1e-9)).T.ravel())
        im = ax.imshow(np.array(mats), aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        bounds = np.cumsum([sum(pattern(r) == k for r in g) for k in order])[:-1]
        for b in bounds:
            ax.axhline(b - 0.5, color='k', lw=0.8)
        for x in [3.5, 7.5]:
            ax.axvline(x, color='k', lw=0.8)
        for x in [1.5, 5.5, 9.5]:
            ax.axvline(x, color='0.5', lw=0.5, ls=':')
        ax.set_xticks(range(12))
        ax.set_xticklabels(['1', '2', '3', '4'] * 3, fontsize=7)
        ax.set_xlabel('location, most contralateral (1) to most ipsilateral (4)', fontsize=7.5)
        for i, e in enumerate(EPOCHS):
            ax.text(1.5 + 4 * i, -0.8, e, ha='center', va='bottom', fontsize=8)
        starts = np.concatenate([[0], bounds])
        for k, s0 in zip(order, starts):
            n = sum(pattern(r) == k for r in g)
            if n:
                ax.text(11.6, s0 + n / 2 - 0.5, f'{k}\n({n})', fontsize=6.5, va='center', ha='left')
        ax.set_title(f'{m}: {"classic SC" if t == "E" else "putative interneurons"} (n={len(g)})', fontsize=10, pad=18)
        ax.set_yticks([])
    plt.subplots_adjust(left=0.02, right=0.9, wspace=0.55, top=0.87, bottom=0.06)
    cax = fig.add_axes([0.975, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cax, label='evoked rate (epoch - baseline) / neuron max |.|')
    fig.suptitle('Responses at the 4 target locations in visual, delay and saccade epochs. Diagonal targets: 1-2 '
                 'contralateral, 3-4 ipsilateral; Newton 08_15: 1 contra horizontal, 2-3 vertical meridian, 4 ipsi.\n'
                 'Rows grouped by which hemifields show significant activation (pipeline tests; vertical meridian '
                 'counts as ipsilateral); within group sorted by visual location ROC.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'location_profiles.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'saved {OUT}/location_profiles.png/.pdf')


if __name__ == '__main__':
    main()
