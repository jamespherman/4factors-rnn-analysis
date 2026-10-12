#!/usr/bin/env python3
"""
Is recurrent connectivity among recorded units structured by factor selectivity?

Core question (AGENTS.md): are W_rec[I -> E] weights structured by the factor
selectivity of the target E units? Analyzed only on models trained with
scripts/train_session.py, where recorded interneurons are fit by I units (see
'recorded_unit_indices' in each config.json). Selectivity is measured on the
recorded neurons (src/analysis/pipeline_selectivity.py, all trials), not on the
model, so it is the same across seeds and variants.

Tests, for each block B in {E->E, I->E, E->I, I->I} among recorded units and each
factor f in {identity, salience, reward, probability}:

    targeting (blocks with E targets): for each presynaptic unit, Spearman rho
        between |w(post, pre)| and the signed selectivity of the postsynaptic E
        neurons; statistic = mean |rho| over presynaptic units (structure in either
        direction). Null: permute selectivity across postsynaptic neurons.
    like-to-like: Pearson r between |w(post, pre)| and -|s_post - s_pre| over
        pairs; positive = stronger connections between similarly selective
        neurons. Null: permute selectivity within each population.
    spatial: mean |w| for pairs sharing a preferred location minus pairs that
        don't. Null: permute preferred locations within each population.

Each statistic is converted to a z-score against its permutation null; sessions
are combined with Stouffer's method. The analysis is repeated for every seed, and
the cross-seed correlation of the recorded-unit weights is reported to show how
well the data constrain them.

Usage:
    python scripts/analyze_ie_structure.py --variant cond_target --seeds 42 43 44
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_mat_file, RNNDataset
from src.analysis.pipeline_selectivity import (
    get_target_onset_bins, compute_window_rates, compute_preferred_locations,
    compute_factor_selectivity
)
from compare_recorded_vs_rnn_selectivity import load_model, SESSIONS, DATA_DIR, RETRAINED_DIR

FACTORS = ['identity', 'salience', 'reward', 'probability']
BLOCKS = {'EE': ('E', 'E'), 'IE': ('I', 'E'), 'EI': ('E', 'I'), 'II': ('I', 'I')}  # (pre, post)
N_PERM = 1000
MIN_NEURONS = 4


def recorded_selectivity(session_id: str) -> dict:
    """Signed visual-epoch selectivity and preferred location of each recorded neuron."""
    data = load_mat_file(str(DATA_DIR / f'rnn_export_Newton_{session_id}_2025_SC.mat'))
    rates = compute_window_rates(np.transpose(data['firing_rates'], (2, 1, 0)), get_target_onset_bins(data))
    preferred = compute_preferred_locations(rates, np.asarray(data['trial_location']).ravel())
    factors = compute_factor_selectivity(rates, data, preferred)
    return {
        'neuron_type': np.asarray(data['neuron_type']).ravel(),
        'selectivity': {f: factors[f]['selectivity'] for f in FACTORS},
        'preferred_location': preferred['for_factors'],
        'data': data,
    }


def perm_z(observed: float, null: np.ndarray) -> float:
    """z-score of observed statistic relative to its permutation null."""
    sd = null.std()
    return float((observed - null.mean()) / sd) if sd > 0 else np.nan


def targeting_stat(W: np.ndarray, s_post: np.ndarray) -> float:
    """Mean |Spearman rho| between each presynaptic unit's |weights| and postsynaptic selectivity."""
    rhos = [stats.spearmanr(np.abs(W[:, j]), s_post)[0] for j in range(W.shape[1])]
    return float(np.nanmean(np.abs(rhos)))


def like_stat(W: np.ndarray, s_post: np.ndarray, s_pre: np.ndarray, offdiag: np.ndarray) -> float:
    """Pearson r between |w| and selectivity similarity over valid pairs."""
    sim = -np.abs(s_post[:, None] - s_pre[None, :])
    return float(np.corrcoef(np.abs(W)[offdiag], sim[offdiag])[0, 1])


def spatial_stat(W: np.ndarray, loc_post: np.ndarray, loc_pre: np.ndarray, offdiag: np.ndarray) -> float:
    """Mean |w| for same-preferred-location pairs minus different-location pairs."""
    same = (loc_post[:, None] == loc_pre[None, :]) & offdiag
    diff = (loc_post[:, None] != loc_pre[None, :]) & offdiag
    if not same.any() or not diff.any():
        return np.nan
    return float(np.abs(W)[same].mean() - np.abs(W)[diff].mean())


def analyze_model(model_dir: Path, rec: dict, rng: np.random.Generator) -> dict:
    """All block x factor statistics for one trained model."""
    dataset = RNNDataset(rec['data'])
    model, config, units = load_model(model_dir, dataset.get_input_dim())
    assert units is not None, f"{model_dir} has no recorded_unit_indices; legacy models can't be analyzed"
    neuron_type = rec['neuron_type']
    assert np.all((units < model.n_exc) == (neuron_type == 1)), "Unit types don't match neuron types"

    W = model.W_rec.detach().cpu().numpy()  # [post, pre]
    idx = {'E': np.where(neuron_type == 1)[0], 'I': np.where(neuron_type == 2)[0]}
    out = {'W_rec_recorded': W[np.ix_(units, units)]}

    for block, (pre, post) in BLOCKS.items():
        res = {}
        for f in FACTORS + ['location']:
            if f == 'location':
                s_post_all = rec['preferred_location'][idx[post]]
                s_pre_all = rec['preferred_location'][idx[pre]]
            else:
                s_post_all = rec['selectivity'][f][idx[post]]
                s_pre_all = rec['selectivity'][f][idx[pre]]
            ok_post = ~np.isnan(s_post_all.astype(float))
            ok_pre = ~np.isnan(s_pre_all.astype(float))
            if ok_post.sum() < MIN_NEURONS or ok_pre.sum() < MIN_NEURONS:
                continue
            post_n, pre_n = idx[post][ok_post], idx[pre][ok_pre]
            Wb = W[np.ix_(units[post_n], units[pre_n])]
            s_post, s_pre = s_post_all[ok_post], s_pre_all[ok_pre]
            offdiag = post_n[:, None] != pre_n[None, :]

            entry = {'n_post': int(len(post_n)), 'n_pre': int(len(pre_n))}
            if f == 'location':
                obs = spatial_stat(Wb, s_post, s_pre, offdiag)
                null = np.array([spatial_stat(Wb, rng.permutation(s_post), rng.permutation(s_pre), offdiag)
                                 for _ in range(N_PERM)])
                entry.update(spatial=obs, spatial_z=perm_z(obs, null[~np.isnan(null)]))
            else:
                obs = like_stat(Wb, s_post, s_pre, offdiag)
                null = np.array([like_stat(Wb, rng.permutation(s_post), rng.permutation(s_pre), offdiag)
                                 for _ in range(N_PERM)])
                entry.update(like=obs, like_z=perm_z(obs, null))
                if post == 'E':
                    obs = targeting_stat(Wb, s_post)
                    null = np.array([targeting_stat(Wb, rng.permutation(s_post)) for _ in range(N_PERM // 4)])
                    entry.update(targeting=obs, targeting_z=perm_z(obs, null))
            res[f] = entry
        out[block] = res
    return out


def stouffer(zs: list) -> tuple:
    """Combined z and two-sided p across sessions."""
    zs = [z for z in zs if z is not None and not np.isnan(z)]
    if not zs:
        return np.nan, np.nan
    z = float(np.sum(zs) / np.sqrt(len(zs)))
    return z, float(2 * stats.norm.sf(abs(z)))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    parser = argparse.ArgumentParser(description='Selectivity structure of recurrent weights')
    parser.add_argument('--variant', default='cond_target', choices=['cond', 'cond_target'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    out_dir = RETRAINED_DIR / 'ie_structure'
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = {s: recorded_selectivity(s) for s in SESSIONS}
    results = {}
    for seed in args.seeds:
        for s in SESSIONS:
            model_dir = RETRAINED_DIR / args.variant / f'Newton_{s}_2025_SC_seed{seed}'
            if not (model_dir / 'model_best.pt').exists():
                print(f"  missing {model_dir}, skipping")
                continue
            print(f"Analyzing {args.variant} seed {seed} session {s}...", flush=True)
            results[(seed, s)] = analyze_model(model_dir, recs[s], rng)

    # Combine sessions per seed with Stouffer's method
    summary = {}
    for seed in args.seeds:
        per_seed = {}
        for block in BLOCKS:
            for f in FACTORS + ['location']:
                for stat in ['targeting_z', 'like_z', 'spatial_z']:
                    zs = [results[(seed, s)][block].get(f, {}).get(stat) for s in SESSIONS if (seed, s) in results]
                    zs = [z for z in zs if z is not None]
                    if zs:
                        z, p = stouffer(zs)
                        per_seed[f'{block}|{f}|{stat[:-2]}'] = {'z': z, 'p': p, 'session_z': zs}
        if per_seed:
            keys_seed = list(per_seed)
            qs = benjamini_hochberg([per_seed[k]['p'] for k in keys_seed])
            for k, q in zip(keys_seed, qs):
                per_seed[k]['q'] = float(q)
        summary[seed] = per_seed

    # Cross-seed consistency of recorded-unit weights, per session and block
    consistency = {}
    for s in SESSIONS:
        nt = recs[s]['neuron_type']
        seeds = [sd for sd in args.seeds if (sd, s) in results]
        for block, (pre, post) in BLOCKS.items():
            post_i, pre_i = np.where(nt == (1 if post == 'E' else 2))[0], np.where(nt == (1 if pre == 'E' else 2))[0]
            rs = []
            for a in range(len(seeds)):
                for b in range(a + 1, len(seeds)):
                    Wa = results[(seeds[a], s)]['W_rec_recorded'][np.ix_(post_i, pre_i)]
                    Wb = results[(seeds[b], s)]['W_rec_recorded'][np.ix_(post_i, pre_i)]
                    off = post_i[:, None] != pre_i[None, :]
                    rs.append(float(np.corrcoef(Wa[off], Wb[off])[0, 1]))
            if rs:
                consistency[f'{s}|{block}'] = float(np.mean(rs))

    with open(out_dir / f'{args.variant}_summary.json', 'w') as f:
        json.dump({'per_seed': {str(k): v for k, v in summary.items()},
                   'cross_seed_weight_correlation': consistency}, f, indent=2)

    # Report
    lines = [f"# Recurrent weight structure vs recorded selectivity ({args.variant})", "",
             f"Seeds: {args.seeds}. Sessions combined per seed with Stouffer's method; "
             f"z-scores against {N_PERM}-permutation nulls (targeting: {N_PERM // 4}).", "",
             "| Block | Factor | Test | " + " | ".join(f"z (seed {sd})" for sd in args.seeds) + " |",
             "|---|---|---|" + "---|" * len(args.seeds)]
    keys = sorted({k for v in summary.values() for k in v},
                  key=lambda k: (list(BLOCKS).index(k.split('|')[0]), (FACTORS + ['location']).index(k.split('|')[1]), k))
    for k in keys:
        block, f, test = k.split('|')
        cells = []
        for sd in args.seeds:
            e = summary[sd].get(k)
            mark = '' if e is None else ('**' if e['q'] < 0.05 else '*' if e['p'] < 0.05 else '')
            cells.append('' if e is None else f"{e['z']:+.2f}{mark}")
        lines.append(f"| {block[0]}→{block[1]} | {f} | {test} | " + " | ".join(cells) + " |")
    lines += ["", "\\* p < 0.05 uncorrected; \\*\\* q < 0.05 after Benjamini-Hochberg across all tests "
              "within a seed.", "","## Cross-seed correlation of recorded-unit weights", "",
              "| Session | E→E | I→E | E→I | I→I |", "|---|---|---|---|---|"]
    for s in SESSIONS:
        lines.append(f"| {s} | " + " | ".join(f"{consistency.get(f'{s}|{b}', np.nan):.2f}" for b in BLOCKS) + " |")
    report = '\n'.join(lines) + '\n'
    (out_dir / f'{args.variant}_report.md').write_text(report, encoding='utf-8')
    print('\n' + report)


if __name__ == '__main__':
    main()
