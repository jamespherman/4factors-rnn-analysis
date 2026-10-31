#!/usr/bin/env python3
"""
Phase 1 / G2 + G3 (plans/2026-10-31_intercollicular_plan.md).

G2, latency vs amplitude: target-aligned PSTHs (1 ms, Gaussian SD 5 ms), evoked = minus the -100..0 ms baseline.
Activation latency = first time >= 30 ms the evoked PSTH reaches half its peak (30-250 ms) and stays there
10 ms; suppression latency likewise for half the trough. Calibration: classic neurons' contralateral activations
over conditions (P and contra-neighbour x high-salience bullseye / low-salience bullseye / image). Question: do
ipsi-location activations (any neuron) sit on the latency-vs-log(amplitude) curve? For suppression: latency at the
ipsi vs the contra 90-deg neighbour (equidistant) within the same neuron.

G3, alignment: for each response (unit x location) the spikes in [target + 50 ms, saccade + 50 ms] are scored
under a target-aligned and a saccade-aligned inhomogeneous Poisson rate estimated on the other half of the trials
(2-fold). Alignment index = (LL_saccade - LL_target) per spike: > 0 saccade-locked. Saccade-aligned onset =
first time the saccade-aligned evoked PSTH reaches half its extreme value.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, mannwhitneyu, linregress

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, OPPOSITE  # noqa: E402

OUT = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g23_timing'
G1 = Path(__file__).parent.parent / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
MIN_TRIALS = 20


def psth(spk, align, trials, t0=-0.1, t1=0.3):
    """Evoked PSTH (sp/s) at 1 ms from t0..t1 s around align[trials], baseline -100..0 ms subtracted."""
    edges = np.arange(t0, t1 + 1e-9, 0.001)
    h = np.zeros(len(edges) - 1)
    for a in align[trials]:
        lo, hi = np.searchsorted(spk, [a + t0, a + t1])
        h += np.histogram(spk[lo:hi] - a, edges)[0]
    r = gaussian_filter1d(h / len(align[trials]) / 0.001, 5)
    tt = edges[:-1] * 1000
    return tt, r - r[(tt >= -100) & (tt < 0)].mean()


def half_latency(tt, r, sign, start=30, stop=250, hold=10):
    sel = (tt >= start) & (tt <= stop)
    x = sign * r
    ext = x[sel].max()
    if ext <= 0:
        return np.nan, 0.0
    above = (x >= ext / 2) & (tt >= start)
    run = np.convolve(above.astype(int), np.ones(hold, int), 'valid') == hold
    idx = np.flatnonzero(run)
    return (float(tt[idx[0]]) if len(idx) else np.nan), float(ext)


def stim_class(t):
    return np.where(t['identity'] < 3, 'image', np.where(t['salience'] == 1, 'bull_hi', 'bull_lo'))


def alignment(t, j, trials):
    """Cross-validated LL difference per spike (saccade- minus target-aligned) and saccade-aligned half-onset."""
    spk = t['spikes'][t['ids'][j]]
    on, sac = t['ev']['targetOn'][trials], t['ev']['saccadeOnset'][trials]
    ok = ~np.isnan(on) & ~np.isnan(sac)
    on, sac = on[ok], sac[ok]
    starts, ends = on + 0.05, sac + 0.05
    spikes = [spk[np.searchsorted(spk, a):np.searchsorted(spk, b)] for a, b in zip(starts, ends)]
    dt = 0.01
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(on)) % 2
    ll = {'target': 0.0, 'saccade': 0.0}
    n_spk = 0
    for k in (0, 1):
        tr, te = fold != k, fold == k
        for name, ref, lo, hi in [('target', on, 0.05, 1.5), ('saccade', sac, -1.5, 0.05)]:
            edges = np.arange(lo, hi + 1e-9, dt)
            cnt = np.zeros(len(edges) - 1)
            cov = np.zeros(len(edges) - 1)
            for i in np.flatnonzero(tr):
                rel = spikes[i] - ref[i]
                cnt += np.histogram(rel, edges)[0]
                a, b = starts[i] - ref[i], ends[i] - ref[i]
                cov += np.clip((np.minimum(edges[1:], b) - np.maximum(edges[:-1], a)) / dt, 0, 1)
            lam = (cnt + 0.5) / (cov * dt + 1e-3)                              # sp/s per bin
            for i in np.flatnonzero(te):
                rel = spikes[i] - ref[i]
                b_idx = np.clip(np.searchsorted(edges, rel) - 1, 0, len(lam) - 1)
                a, b = starts[i] - ref[i], ends[i] - ref[i]
                expo = np.clip((np.minimum(edges[1:], b) - np.maximum(edges[:-1], a)), 0, None)
                ll[name] += np.log(lam[b_idx]).sum() - (lam * expo).sum()
        n_spk += sum(len(spikes[i]) for i in np.flatnonzero(fold == k))
    tt, r = psth(spk, t['ev']['saccadeOnset'], trials, -0.3, 0.1)
    base = psth(spk, t['ev']['targetOn'], trials)[1]
    sign = 1 if r[(tt >= -100) & (tt <= 50)].mean() >= 0 else -1
    onset, _ = half_latency(tt, r, sign, start=-250, stop=50, hold=10)
    return (ll['saccade'] - ll['target']) / max(n_spk, 1), onset


def main():
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    lat_rows, sup_rows, al_rows = [], [], []
    for s in ALL_SESSIONS:
        if s == '08_15':
            continue
        t = session_table(s)
        sc = stim_class(t)
        contra = [k + 1 for k in range(4) if t['hemi'][k] == 'contra']
        for j, i in enumerate(t['ids']):
            r1 = g1[(s, i)]
            P = r1['P']
            cn = [k for k in contra if k != P][0]
            inb = [k for k in range(1, 5) if t['hemi'][k - 1] == 'ipsi' and t['elev'][k - 1] == t['elev'][P - 1]][0]
            op = OPPOSITE[P]
            spk = t['spikes'][i]
            st = t['stable'][:, j]
            for role, k in [('P', P), ('cn', cn), ('in', inb), ('op', op)]:
                # G2 activation latencies by stimulus class
                for c in ('bull_hi', 'bull_lo', 'image'):
                    tr = st & (t['loc'] == k) & (sc == c)
                    if tr.sum() < MIN_TRIALS:
                        continue
                    tt, r = psth(spk, t['ev']['targetOn'], tr)
                    lat, amp = half_latency(tt, r, 1)
                    sig = r1[f'visual_{role}'] > 0 and r1[f'visual_{role}_p'] < 0.01
                    if sig and amp >= 5 and not np.isnan(lat):
                        lat_rows.append({'session': s, 'id': i, 'type': t['type'][j], 'role': role, 'cls': c,
                                         'lat': lat, 'amp': amp})
                tr = st & (t['loc'] == k)
                # G2 suppression latency (all trials at the location)
                if role in ('cn', 'in', 'op') and r1[f'visual_{role}'] < 0 and r1[f'visual_{role}_p'] < 0.01:
                    tt, r = psth(spk, t['ev']['targetOn'], tr)
                    lat, depth = half_latency(tt, r, -1)
                    sup_rows.append({'session': s, 'id': i, 'type': t['type'][j], 'role': role, 'lat': lat, 'depth': depth})
                # G3 alignment for every significant response at this location
                sig_any = any(r1[f'{w}_{role}_p'] < 0.01 for w in ('visual', 'presac'))
                if sig_any and tr.sum() >= MIN_TRIALS:
                    ai, onset = alignment(t, j, tr)
                    sign = np.sign(r1[f'presac_{role}'] if r1[f'presac_{role}_p'] < 0.01 else r1[f'visual_{role}'])
                    al_rows.append({'session': s, 'id': i, 'type': t['type'][j], 'role': role, 'ai': ai,
                                    'onset': onset, 'sign': sign})
        print(s, 'done', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'rows.npy', {'lat': lat_rows, 'sup': sup_rows, 'al': al_rows}, allow_pickle=True)
    report(lat_rows, sup_rows, al_rows)


def report(lat_rows, sup_rows, al_rows):
    L = []
    p = L.append
    p('# G2 latency vs amplitude, G3 target vs saccade alignment (diagonal-target sessions)\n')
    cal = [r for r in lat_rows if r['type'] == 'E' and r['role'] in ('P', 'cn')]
    x, y = np.log(np.array([r['amp'] for r in cal])), np.array([r['lat'] for r in cal])
    fit = linregress(x, y)
    p(f'Calibration (classic contra, n={len(cal)} conditions): latency = {fit.intercept:.1f} + {fit.slope:.1f}*ln(amp) ms, '
      f'r={fit.rvalue:.2f}; residual SD {np.std(y - fit.intercept - fit.slope * x):.1f} ms')
    def resid(g):
        return np.array([r['lat'] - fit.intercept - fit.slope * np.log(r['amp']) for r in g])
    for t in ('E', 'I'):
        for roles, label in [(('P', 'cn'), 'contra'), (('in', 'op'), 'ipsi')]:
            g = [r for r in lat_rows if r['type'] == t and r['role'] in roles]
            if len(g) >= 3:
                res = resid(g)
                p(f'- {t} {label:6s} activations: n={len(g):3d} conditions from {len({(r["session"], r["id"]) for r in g})} '
                  f'units, median amp {np.median([r["amp"] for r in g]):.1f} sp/s, median latency {np.median([r["lat"] for r in g]):.0f} ms, '
                  f'residual median {np.median(res):+.1f} ms (IQR {np.percentile(res, 25):+.1f} to {np.percentile(res, 75):+.1f})')
    ipsiI = resid([r for r in lat_rows if r['type'] == 'I' and r['role'] in ('in', 'op')])
    contraI = resid([r for r in lat_rows if r['type'] == 'I' and r['role'] in ('P', 'cn')])
    if len(ipsiI) and len(contraI):
        p(f'  I ipsi vs I contra residuals: Mann-Whitney p={mannwhitneyu(ipsiI, contraI).pvalue:.2g}')
    # Within-unit: ipsi activation latency minus contra-neighbour/P latency for the same class of stimulus
    pairs = []
    by = {}
    for r in lat_rows:
        by.setdefault((r['session'], r['id'], r['cls']), {})[r['role']] = r
    for k, d in by.items():
        if 'in' in d and ('cn' in d or 'P' in d):
            c = d.get('cn', d.get('P'))
            pairs.append((d['in']['lat'] - c['lat'], np.log(d['in']['amp'] / c['amp'])))
    if len(pairs) >= 5:
        dl, da = np.array(pairs).T
        p(f'- Within unit, ipsi-neighbour minus contra latency: n={len(dl)}, median {np.median(dl):+.1f} ms '
          f'(Wilcoxon p={wilcoxon(dl).pvalue:.2g}); expected from amplitude difference '
          f'{np.median(fit.slope * da):+.1f} ms; excess {np.median(dl - fit.slope * da):+.1f} ms '
          f'(p={wilcoxon(dl - fit.slope * da).pvalue:.2g})')
    p('\n## Suppression latency (visual window), equidistant neighbours within unit')
    by = {}
    for r in sup_rows:
        by.setdefault((r['session'], r['id']), {})[r['role']] = r
    d = [(v['in']['lat'] - v['cn']['lat']) for v in by.values() if 'in' in v and 'cn' in v
         and not np.isnan(v['in']['lat']) and not np.isnan(v['cn']['lat'])]
    for role in ('cn', 'in', 'op'):
        g = [r['lat'] for r in sup_rows if r['role'] == role and not np.isnan(r['lat'])]
        if g:
            p(f'- {role}: n={len(g)}, median half-trough latency {np.median(g):.0f} ms')
    if len(d) >= 3:
        p(f'- within unit ipsi-nb minus contra-nb: n={len(d)}, median {np.median(d):+.0f} ms, Wilcoxon p={wilcoxon(d).pvalue:.2g}')
    p('\n## G3 alignment index ((LL_sac - LL_target)/spike; > 0 saccade-locked) and saccade-aligned half-onset')
    for t in ('E', 'I'):
        for roles, label in [(('P',), 'preferred'), (('cn',), 'contra-nb'), (('in',), 'ipsi-nb'), (('op',), 'opposite')]:
            for sg, sl in [(1, 'activ'), (-1, 'suppr')]:
                g = [r for r in al_rows if r['type'] == t and r['role'] in roles and r['sign'] == sg]
                if len(g) >= 5:
                    ai = np.array([r['ai'] for r in g])
                    on = np.array([r['onset'] for r in g])
                    p(f'- {t} {label:9s} {sl}: n={len(g):3d}, AI median {np.median(ai):+.4f}, saccade-locked {np.mean(ai > 0):.0%}, '
                      f'sacc-aligned onset median {np.nanmedian(on):+.0f} ms')
    text = '\n'.join(L)
    print(text)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
