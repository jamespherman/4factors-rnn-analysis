#!/usr/bin/env python3
"""
Stage A of specs/CONTEXT_INPUTS_PLAN.md: data gates before training context models.

A0. Context labels: label-derived reward map / block vs raw trialInfo.
A1. Context effects that survive slow drift. Block context is confounded with time
    in session (6-14 reward-map switches, 3-8 block switches per session), and
    random train/held-out splits share drift, so:
    a. Circular-shift null for the pipeline's visual-epoch factor selectivity
       (labels shifted against rates, which keeps block run structure).
    b. Switch-triggered contrasts: K trials before vs after each context switch,
       signed by whether the neuron's preferred location gains value. The
       contingency alternates across switches; drift doesn't. Effective N = switches.
    c. Probability sign reversal: block effect at location 1 vs location 3 (opposite
       signs for a probability effect, same sign for drift).
    d. Trial-history regression: visual response at the preferred location vs
       recent target locations (raw trial sequence), current context and drift.
A2. Population decoding of context and factors from recorded I neurons vs
    firing-rate-matched E neurons. Context decoders are cross-validated across
    reward-map runs and compared with a circular-shift null.

Outputs: results/context_analysis/stage_a_report.md and stage_a_results.json

Usage:
    python scripts/analyze_context_effects.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_mat_file
from src.sessions import DEFAULT_SESSIONS, ALL_SESSIONS, session_name, load_export
from src.analysis.pipeline_selectivity import (
    get_target_onset_bins, compute_window_rates, compute_preferred_locations, get_factor_masks
)
from src.analysis.roc_selectivity import compute_roc_auc
from src.analysis.task_context import (
    get_task_context, location_rewarded, location_frequent, get_switches, match_raw_trials
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'results' / 'context_analysis'
SESSIONS = DEFAULT_SESSIONS

PRE_WINDOW_MS = (-300.0, 0.0)
SWITCH_K = (15, 30)
N_SHIFTS = 200
N_DECODE_SHIFTS = 50
N_E_DRAWS = 30
RNG = np.random.default_rng(0)


def load_session(session_id: str, trials: str = 'all') -> dict:
    """
    Load export, window rates, context and preferred locations for one session.

    trials: 'all' exported trials, or 'verified' (successful trials only; see
        src/sessions.py)
    """
    return session_from_data(load_export(session_id, trials), session_id)


def session_from_data(data: dict, session_id: str) -> dict:
    """Window rates, context and preferred locations for a loaded (or synthetic) export."""
    name = session_name(session_id)
    rates = np.transpose(data['firing_rates'], (2, 1, 0))
    onset = get_target_onset_bins(data)
    vis = compute_window_rates(rates, onset)
    pre = compute_window_rates(rates, onset, window_ms=PRE_WINDOW_MS)
    location = np.asarray(data['trial_location']).ravel().astype(int)
    return {
        'id': session_id, 'name': name, 'data': data, 'vis': vis, 'pre': pre,
        'location': location, 'context': get_task_context(data),
        'preferred': compute_preferred_locations(vis, location),
        'is_e': np.asarray(data['neuron_type']).ravel() == 1,
    }


def roc_sel(x1: np.ndarray, x2: np.ndarray, min_n: int = 5) -> float:
    """2 * (AUC - 0.5), AUC > 0.5 when x2 is higher; NaN if too few values."""
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1) < min_n or len(x2) < min_n:
        return np.nan
    return 2 * (compute_roc_auc(x1, x2) - 0.5)


# ---------------------------------------------------------------- A0

def check_labels(sess: dict) -> dict:
    """Agreement of label-derived context with raw trialInfo."""
    raw = match_raw_trials(sess['data'], sess['name'])
    if raw is None:
        return {'matched': False}
    idx = raw['export_to_raw']
    ctx = sess['context']
    hb_odd = np.isin(raw['halfBlock'][idx], [1, 3]).astype(int)
    ok = ~np.isnan(ctx['block'])
    return {
        'matched': True,
        'reward_map_agreement': float(np.mean(ctx['reward_map'] == hb_odd)),
        'block_agreement': float(np.mean(ctx['block'][ok] == raw['blockNumber'][idx][ok])),
        'n_reward_switches': int(len(get_switches(ctx['reward_map']))),
        'n_block_switches': int(len(get_switches(ctx['block']))),
        'reward_run_lengths': np.diff(np.r_[0, get_switches(ctx['reward_map']), len(idx)]).tolist(),
    }


# ---------------------------------------------------------------- A1a

def circular_null_selectivity(sess: dict) -> dict:
    """Per-neuron p-values of pipeline factor selectivity against circular-shift nulls."""
    data, vis, location = sess['data'], sess['vis'], sess['location']
    n_trials = len(location)
    masks = get_factor_masks(data)
    shifts = RNG.integers(n_trials // 10, n_trials - n_trials // 10, N_SHIFTS)
    out = {}
    for factor, (c1, c2) in masks.items():
        key = 'for_probability' if factor == 'probability' else 'for_factors'
        pvals = np.full(vis.shape[1], np.nan)
        for i in range(vis.shape[1]):
            pref = sess['preferred'][key][i]

            def sel(a, b, loc):
                m = loc == pref
                return roc_sel(vis[m & a, i], vis[m & b, i])

            obs = sel(c1, c2, location)
            if np.isnan(obs):
                continue
            null = np.array([sel(np.roll(c1, k), np.roll(c2, k), np.roll(location, k)) for k in shifts])
            null = null[~np.isnan(null)]
            pvals[i] = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))
        out[factor] = pvals
    return out


# ---------------------------------------------------------------- A1b

def switch_contrasts(sess: dict, K: int) -> list:
    """
    Signed before/after-switch differences per (context, epoch, switch, neuron).

    Sign is +1 if the neuron's preferred location gains value at the switch (becomes
    rewarded / frequent). Pre-target epoch uses all trials; visual epoch uses trials
    with the target at the preferred location.
    """
    ctx, location = sess['context'], sess['location']
    n_trials = len(location)
    rows = []
    for ctx_name, state in [('reward', ctx['reward_map'].astype(float)), ('block', ctx['block'])]:
        switches = get_switches(state)
        for i in range(sess['vis'].shape[1]):
            pref = sess['preferred']['for_factors'][i]
            if ctx_name == 'block':
                pref = sess['preferred']['for_probability'][i]
            gains = (location_rewarded(state, pref) if ctx_name == 'reward'
                     else location_frequent(state, pref))
            for epoch, rates, loc_mask in [('pre', sess['pre'], np.ones(n_trials, bool)),
                                           ('vis', sess['vis'], location == pref)]:
                for k, t in enumerate(switches):
                    lo = max(t - K, switches[k - 1] if k > 0 else 0)
                    hi = min(t + K, switches[k + 1] if k + 1 < len(switches) else n_trials)
                    before, after = np.arange(lo, t), np.arange(t, hi)
                    before = before[loc_mask[before] & ~np.isnan(rates[before, i])]
                    after = after[loc_mask[after] & ~np.isnan(rates[after, i])]
                    if len(before) < 3 or len(after) < 3:
                        continue
                    sign = 1.0 if gains[t] else -1.0
                    rows.append({'context': ctx_name, 'epoch': epoch, 'session': sess['id'],
                                 'switch': int(t), 'neuron': i, 'is_e': bool(sess['is_e'][i]),
                                 'diff': sign * (rates[after, i].mean() - rates[before, i].mean())})
    return rows


def summarize_switches(rows: list) -> dict:
    """Population tests: across switches (primary; neuron-averaged per switch) and across neurons."""
    out = {}
    for ctx_name in ('reward', 'block'):
        for epoch in ('pre', 'vis'):
            for cell in ('all', 'E', 'I'):
                sub = [r for r in rows if r['context'] == ctx_name and r['epoch'] == epoch
                       and (cell == 'all' or r['is_e'] == (cell == 'E'))]
                if not sub:
                    continue
                per_switch = {}
                per_neuron = {}
                for r in sub:
                    per_switch.setdefault((r['session'], r['switch']), []).append(r['diff'])
                    per_neuron.setdefault((r['session'], r['neuron']), []).append(r['diff'])
                sw = np.array([np.mean(v) for v in per_switch.values()])
                nr = np.array([np.mean(v) for v in per_neuron.values()])
                entry = {'n_switches': len(sw), 'n_neurons': len(nr),
                         'mean_diff_sp_s': float(sw.mean()),
                         'frac_neurons_positive': float(np.mean(nr > 0))}
                if len(sw) >= 3:
                    entry['p_switches'] = float(ttest_1samp(sw, 0).pvalue)
                if len(nr) >= 5:
                    entry['p_neurons'] = float(wilcoxon(nr).pvalue)
                out[f'{ctx_name}|{epoch}|{cell}'] = entry
    return out


# ---------------------------------------------------------------- A1c

def probability_sign_reversal(sess: dict, min_trials: int = 5) -> list:
    """
    Per neuron: block-1-minus-block-2 visual response at location 1 (d1) and at
    location 3 (d3). Probability predicts d1 > 0 > d3 (or the reverse for an
    adaptation effect); drift shifts both the same way. Returns (d1, d3, is_e).
    """
    out = []
    block, location, vis = sess['context']['block'], sess['location'], sess['vis']
    for i in range(vis.shape[1]):
        d = []
        for loc in (1, 3):
            a = vis[(location == loc) & (block == 1), i]
            b = vis[(location == loc) & (block == 2), i]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            d.append(a.mean() - b.mean() if len(a) >= min_trials and len(b) >= min_trials else np.nan)
        out.append((d[0], d[1], bool(sess['is_e'][i])))
    return out


# ---------------------------------------------------------------- A1d

def history_regression(sess: dict) -> dict:
    """
    OLS per neuron on visual responses at the preferred location:
        rate ~ same_as_prev + n_same_in_prev3 + frequent + rewarded + stimulus class
               + trial index + trial index^2
    Trial history from the raw trial sequence (previous trials with an assigned
    target location, whatever their outcome). Returns t-values per regressor.
    """
    raw = match_raw_trials(sess['data'], sess['name'])
    if raw is None:
        return {}
    raw_loc = raw['targetLocIdx']
    export_to_raw = raw['export_to_raw']
    location, vis, ctx, data = sess['location'], sess['vis'], sess['context'], sess['data']
    n_trials = len(location)

    prev = np.full((n_trials, 3), np.nan)
    has_loc = np.flatnonzero(~np.isnan(raw_loc))
    for t, r in enumerate(export_to_raw):
        earlier = has_loc[has_loc < r][-3:][::-1]
        prev[t, :len(earlier)] = raw_loc[earlier]

    identity = np.asarray(data['trial_identity']).ravel()
    salience = np.asarray(data['trial_salience']).ravel()
    trial_z = (np.arange(n_trials) - n_trials / 2) / (n_trials / 2)
    names = ['same_as_prev', 'n_same_prev3', 'frequent', 'rewarded', 'trial', 'trial_sq']
    tvals = {n: [] for n in names}
    for i in range(vis.shape[1]):
        pref = sess['preferred']['for_probability'][i]  # location 1 or 3, so 'frequent' is defined
        m = (location == pref) & ~np.isnan(vis[:, i]) & ~np.isnan(prev).any(axis=1)
        if m.sum() < 30:
            continue
        X = np.column_stack([
            prev[m, 0] == pref, (prev[m] == pref).sum(axis=1),
            location_frequent(ctx['block'], pref)[m], location_rewarded(ctx['reward_map'], pref)[m],
            trial_z[m], trial_z[m] ** 2,
            identity[m] == 1, identity[m] == 2, salience[m] == 1,
            np.ones(m.sum()),
        ]).astype(float)
        y = vis[m, i]
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < X.shape[1]:
            continue
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        cov = np.linalg.inv(X.T @ X) * (resid @ resid / dof)
        t = beta / np.sqrt(np.diag(cov))
        for j, n in enumerate(names):
            tvals[n].append(float(t[j]))
    return tvals


# ---------------------------------------------------------------- A2

def decode(X: np.ndarray, y: np.ndarray, groups: np.ndarray = None) -> float:
    """Cross-validated balanced accuracy (logistic regression)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000,
                                                             class_weight='balanced'))
    if groups is not None:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        pred = cross_val_predict(clf, X, y, cv=cv)
    return balanced_accuracy_score(y, pred)


def rate_matched_e_subsets(sess: dict, rates: np.ndarray, n_draws: int) -> list:
    """E-neuron subsets the size of the I population, matched to I firing rates."""
    e_idx, i_idx = np.flatnonzero(sess['is_e']), np.flatnonzero(~sess['is_e'])
    mean_rate = np.nanmean(rates, axis=0)
    subsets = []
    for _ in range(n_draws):
        available = list(e_idx)
        chosen = []
        for i in RNG.permutation(i_idx):
            nearest = sorted(available, key=lambda e: abs(mean_rate[e] - mean_rate[i]))[:3]
            pick = nearest[RNG.integers(len(nearest))]
            chosen.append(pick)
            available.remove(pick)
        subsets.append(np.array(chosen))
    return subsets


def decoding_analysis(sess: dict) -> dict:
    """Decoding accuracy from I neurons vs rate-matched E subsets, per target variable."""
    data, ctx, location = sess['data'], sess['context'], sess['location']
    identity = np.asarray(data['trial_identity']).ravel()
    salience = np.asarray(data['trial_salience']).ravel()
    i_idx = np.flatnonzero(~sess['is_e'])
    n_trials = len(location)

    targets = {
        # name: (epoch, trial mask, labels, grouped CV for context variables)
        'reward_map_pre': ('pre', np.ones(n_trials, bool), ctx['reward_map'], True),
        'block_pre': ('pre', ~np.isnan(ctx['block']), ctx['block'], True),
        'location_vis': ('vis', np.ones(n_trials, bool), location, False),
        'identity_vis': ('vis', np.isin(identity, [1, 2]), identity, False),
        'salience_vis': ('vis', identity == 3, salience, False),
    }
    out = {}
    for name, (epoch, mask, labels, grouped) in targets.items():
        rates = sess[epoch]
        valid = mask & ~np.isnan(rates).any(axis=1)
        y = labels[valid].astype(int)
        groups = ctx['reward_run'][valid] if grouped else None
        if len(np.unique(y)) < 2 or valid.sum() < 40:
            continue

        def acc(cols, yy=y):
            return decode(rates[valid][:, cols], yy, groups)

        entry = {'n_trials': int(valid.sum()), 'n_i': len(i_idx), 'acc_i': acc(i_idx),
                 'acc_e_matched': [acc(c) for c in rate_matched_e_subsets(sess, rates[valid], N_E_DRAWS)]}
        if grouped:
            # Circular-shift null: same CV, labels shifted against rates (keeps run structure)
            full_y = labels.astype(float)
            null = []
            for k in RNG.integers(n_trials // 10, n_trials - n_trials // 10, N_DECODE_SHIFTS):
                yy = np.roll(full_y, k)[valid]
                ok = ~np.isnan(yy)
                if len(np.unique(yy[ok])) < 2:
                    continue
                null.append(decode(rates[valid][ok][:, i_idx], yy[ok].astype(int),
                                   np.roll(ctx['reward_run'], k)[valid][ok]))
            entry['null_i'] = null
            entry['p_i_vs_null'] = float((1 + np.sum(np.array(null) >= entry['acc_i'])) / (1 + len(null)))
        out[name] = entry
    return out


# ---------------------------------------------------------------- report

def main():
    import argparse
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description='Stage A context analyses')
    parser.add_argument('--trials', default='all', choices=['all', 'verified'],
                        help="'verified': successful trials with a verified PLDAPS match only")
    parser.add_argument('--sessions', default=None,
                        help="Comma-separated sessions, or 'all' for every corrected export (default: the Newton three)")
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    elif args.trials == 'verified':
        OUTPUT_DIR = OUTPUT_DIR.parent / 'context_analysis_verified'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = (ALL_SESSIONS if args.sessions == 'all' else args.sessions.split(',')) if args.sessions else SESSIONS
    sessions = [load_session(s, args.trials) for s in names]
    results = {'labels': {}, 'circular_null': {}, 'switch': {}, 'sign_reversal': [],
               'history': {}, 'decoding': {}}
    lines = ['# Stage A: context effects in the recordings', '',
             'See `specs/CONTEXT_INPUTS_PLAN.md` §3. Generated by `scripts/analyze_context_effects.py`.', '']

    # A0
    lines += ['## A0. Context labels vs raw trialInfo', '',
              '| Session | Reward map agreement | Block agreement | Reward switches | Block switches |',
              '|---|---|---|---|---|']
    for sess in sessions:
        r = check_labels(sess)
        results['labels'][sess['id']] = r
        if r['matched']:
            lines.append(f"| {sess['id']} | {r['reward_map_agreement']:.3f} | {r['block_agreement']:.3f} | "
                         f"{r['n_reward_switches']} | {r['n_block_switches']} |")
        else:
            lines.append(f"| {sess['id']} | not matched | | | |")

    # A1a
    print('A1a circular-shift null...')
    lines += ['', '## A1a. Visual-epoch factor selectivity vs circular-shift null', '',
              'Fraction of neurons with p < 0.05 (expected 0.05 under the null).', '',
              '| Factor | All | E | I | n |', '|---|---|---|---|---|']
    pooled = {}
    for sess in sessions:
        for f, p in circular_null_selectivity(sess).items():
            pooled.setdefault(f, []).append((p, sess['is_e']))
    for f, parts in pooled.items():
        p = np.concatenate([x[0] for x in parts])
        e = np.concatenate([x[1] for x in parts])
        ok = ~np.isnan(p)
        frac = lambda m: float(np.mean(p[ok & m] < 0.05)) if (ok & m).any() else np.nan
        results['circular_null'][f] = {'frac_all': frac(ok), 'frac_e': frac(e), 'frac_i': frac(~e),
                                       'n': int(ok.sum())}
        lines.append(f"| {f} | {frac(ok):.2f} | {frac(e):.2f} | {frac(~e):.2f} | {ok.sum()} |")

    # A1b
    print('A1b switch-triggered contrasts...')
    lines += ['', '## A1b. Switch-triggered context effects', '',
              'Mean signed after-minus-before difference (sp/s; positive = higher when the '
              "neuron's preferred location gains value). p across switches (t-test on "
              'neuron-averaged differences; primary) and across neurons (Wilcoxon).', '',
              '| K | Context | Epoch | Cells | Switches | Mean diff | p (switches) | Neurons | Frac > 0 | p (neurons) |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for K in SWITCH_K:
        rows = [r for sess in sessions for r in switch_contrasts(sess, K)]
        summ = summarize_switches(rows)
        results['switch'][K] = summ
        for key, e in summ.items():
            c, ep, cell = key.split('|')
            lines.append(f"| {K} | {c} | {ep} | {cell} | {e['n_switches']} | {e['mean_diff_sp_s']:+.2f} | "
                         f"{e.get('p_switches', np.nan):.2g} | {e['n_neurons']} | "
                         f"{e['frac_neurons_positive']:.2f} | {e.get('p_neurons', np.nan):.2g} |")

    # A1c
    print('A1c probability sign reversal...')
    sr = np.array([x for sess in sessions for x in probability_sign_reversal(sess)], dtype=float)
    ok = ~np.isnan(sr[:, 0]) & ~np.isnan(sr[:, 1])
    d1, d3 = sr[ok, 0], sr[ok, 1]
    results['sign_reversal'] = {'n': int(ok.sum()), 'mean_d1': float(d1.mean()), 'mean_d3': float(d3.mean()),
                                'p_prob_component': float(wilcoxon(d1 - d3).pvalue),
                                'p_common_component': float(wilcoxon(d1 + d3).pvalue)}
    lines += ['', '## A1c. Probability sign reversal', '',
              'Per neuron, block 1 minus block 2 visual response at location 1 (d1) and '
              'location 3 (d3). Location 1 is frequent in block 1 and location 3 in block 2, so a '
              'probability effect gives d1 and d3 opposite signs (tested by d1 - d3); drift '
              'shifts both the same way (d1 + d3).', '',
              f"n = {ok.sum()} neurons; mean d1 = {d1.mean():+.2f}, mean d3 = {d3.mean():+.2f} sp/s; "
              f"probability component (d1 - d3)/2 = {(d1 - d3).mean() / 2:+.2f} sp/s, "
              f"p = {results['sign_reversal']['p_prob_component']:.2g}; common component "
              f"(d1 + d3)/2 = {(d1 + d3).mean() / 2:+.2f} sp/s, p = {results['sign_reversal']['p_common_component']:.2g}. "
              'Positive probability component = higher response when the location is frequent.']

    # A1d
    print('A1d history regression...')
    lines += ['', '## A1d. Trial history vs context (visual response at preferred location 1 or 3)', '',
              '| Regressor | Neurons | Mean t | Frac t > 2 | Frac t < -2 |', '|---|---|---|---|---|']
    pooled_t = {}
    for sess in sessions:
        for n, t in history_regression(sess).items():
            pooled_t.setdefault(n, []).extend(t)
    for n, t in pooled_t.items():
        t = np.array(t)
        results['history'][n] = {'n': len(t), 'mean_t': float(t.mean()), 'frac_pos': float(np.mean(t > 2)),
                                 'frac_neg': float(np.mean(t < -2))}
        lines.append(f"| {n} | {len(t)} | {t.mean():+.2f} | {np.mean(t > 2):.2f} | {np.mean(t < -2):.2f} |")

    # A2
    print('A2 decoding...')
    lines += ['', '## A2. Decoding from recorded I neurons vs rate-matched E neurons', '',
              'Balanced accuracy. Context variables use the pre-target epoch, CV across reward-map '
              f'runs, and a circular-shift null for I ({N_DECODE_SHIFTS} shifts). E: median '
              f'[5th-95th percentile] over {N_E_DRAWS} rate-matched subsets of the I count.', '',
              '| Session | Target | n I | I | Null I (mean) | p I vs null | E matched |',
              '|---|---|---|---|---|---|---|']
    for sess in sessions:
        dec = decoding_analysis(sess)
        results['decoding'][sess['id']] = dec
        for name, e in dec.items():
            ea = np.array(e['acc_e_matched'])
            null = f"{np.mean(e['null_i']):.2f}" if 'null_i' in e else '-'
            p = f"{e['p_i_vs_null']:.2g}" if 'p_i_vs_null' in e else '-'
            lines.append(f"| {sess['id']} | {name} | {e['n_i']} | {e['acc_i']:.2f} | {null} | {p} | "
                         f"{np.median(ea):.2f} [{np.percentile(ea, 5):.2f}-{np.percentile(ea, 95):.2f}] |")

    report = '\n'.join(lines) + '\n'
    (OUTPUT_DIR / 'stage_a_report.md').write_text(report, encoding='utf-8')
    with open(OUTPUT_DIR / 'stage_a_results.json', 'w') as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    sys.stdout.reconfigure(errors='replace')
    print('\n' + report)


if __name__ == '__main__':
    main()
