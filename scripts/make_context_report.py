#!/usr/bin/env python3
"""
Build results/context_report_v12.html: a self-contained report on reward-context inputs
on the corrected exports (pipeline 1.2.0, 7 sessions). It covers the population effect
(results/context_analysis_v12), the context grid with an interleaved split
(results/grid_v12) and with whole-switch held-out windows (results/grid_v12s), the
recovery test with idealized teachers (results/recovery_v12) and with teachers given the
recordings' slow fluctuations (results/recovery_v12_drift), and the fit-quality figures
(results/psth_match, results/r01_figure). Parts whose results are missing are marked as
pending, so the report can be rebuilt as runs finish.

Usage:
    python scripts/make_context_report.py
"""

import base64
import io
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

import make_recovery_report as R
from make_recovery_report import Plot, esc, fmt, mean, table, strip_by_variant, fig_models, CSS
from summarize_grid_by_session import heldout
from src.sessions import ALL_SESSIONS, load_export, stable_mask
from src.analysis.task_context import get_task_context, get_switches

ROOT = Path(__file__).parent.parent
RES = ROOT / 'results'
OUT = RES / 'context_report_v12.html'
NEWTON_EFFECT = ('08_13', '08_14')  # sessions with a population reward-context effect

R.VARIANTS['shared_free_qshift'] = ('Shifted shared → all', 'none', 'control: context shifted out of register')
GRID_ORDER = ['none_free', 'shared_free', 'sharedI_free', 'freeI_free', 'freeE_free', 'free_free',
              'shared_free_qshift', 'free_free_qshift', 'freeI_free_qshift']
STUDENTS = ['none_free', 'shared_free', 'sharedI_free', 'freeI_free', 'free_free']
TEACHERS = {'null': ('No context', 'none'), 'shared': ('Shared → all', 'shared'),
            'ext': ('Per-unit → all', 'unit'), 'local': ('Per-unit → I', 'unit')}


def label(s: str) -> str:
    return s.replace('Feynman_', 'F ') if s.startswith('Feynman') else f'N {s}'


# ---------------------------------------------------------------- data

def load_raw(root: Path):
    f = root / 'evaluation_raw.npy'
    return np.load(f, allow_pickle=True).item() if f.exists() else None


def grid_stats(raw: dict, sessions: list, model_on_test: bool, key: str = 'reward_pre') -> tuple:
    """{variant: [r per restart]} pooled over sessions, and the ceiling per restart."""
    variants = sorted({k.split('/')[0] for k in raw})
    seeds = sorted({k.rsplit('_seed', 1)[1] for k in raw}, key=int)
    out, ceil = {}, {}
    for v in variants:
        vals = []
        for sd in seeds:
            runs = [raw.get(f'{v}/{s}_seed{sd}') for s in sessions]
            if any(r is None for r in runs):
                continue
            r, c = heldout(runs, key, model_on_test)
            vals.append(r)
            ceil[sd] = c
        out[v] = vals
    return out, list(ceil.values())


def recovery_stats(root: Path) -> dict:
    """Per teacher: student r per restart (pooled over sessions, model on test), ceiling, shared - free."""
    mf = root / 'manifest.json'
    if not mf.exists():
        return {}
    manifest = json.loads(mf.read_text())
    out = {}
    for t in TEACHERS:
        raws = {s: load_raw(root / f'{t}_{s}') for s in manifest}
        if any(r is None for r in raws.values()):
            continue
        seeds = sorted({k.rsplit('_seed', 1)[1] for r in raws.values() for k in r}, key=int)
        st, ceil = {v: [] for v in STUDENTS}, []
        for sd in seeds:
            for v in STUDENTS:
                runs = [raws[s].get(f'{v}/{s}_seed{sd}') for s in manifest]
                if any(r is None for r in runs):
                    continue
                r, c = heldout(runs, 'reward_pre', True)
                st[v].append(r)
            ceil.append(c)
        diff = [a - b for a, b in zip(st['shared_free'], st['free_free'])]
        # Held-out LL, shared - free (x1000); nanmean skips neurons silent in the test windows
        ll_diff = [1000 * (np.nanmean(np.concatenate([raws[s][f'shared_free/{s}_seed{sd}']['ll'] for s in manifest]))
                           - np.nanmean(np.concatenate([raws[s][f'free_free/{s}_seed{sd}']['ll'] for s in manifest])))
                   for sd in seeds]
        out[t] = {'students': st, 'ceiling': mean(ceil), 'diff': diff, 'll_diff': ll_diff,
                  'drift_amp': next(iter(manifest.values())).get('drift_amp', {})}
    return out


def stage_a_sessions() -> list:
    """Per-session reward and block pre-target effects from the Stage A note."""
    rows = []
    for line in (RES / 'context_analysis_v12' / 'NOTE.md').read_text(encoding='utf-8').splitlines():
        m = re.search(r'^\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|\s*([+-]?[\d.]+)\s*\(p ([\d.]+)\)\s*\|\s*([+-]?[\d.]+)\s*\(p ([\d.]+)\)', line)
        if m:
            s = m.group(1).replace('Newton_', '')
            rows.append({'session': s, 'n': int(m.group(2)), 'reward': float(m.group(3)), 'p_reward': float(m.group(4)),
                         'block': float(m.group(5)), 'p_block': float(m.group(6))})
    return rows


def gather() -> dict:
    d = {'counts': {}}
    for s in ALL_SESSIONS:
        data = load_export(s, 'verified')
        nt = np.asarray(data['neuron_type']).ravel()
        d['counts'][s] = {'trials': int(data['firing_rates'].shape[2]), 'E': int((nt == 1).sum()), 'I': int((nt == 2).sum()),
                          'switches': len(get_switches(get_task_context(data)['reward_map'])),
                          'stable': float(stable_mask(data).mean())}
    stage = json.loads((RES / 'context_analysis_v12' / 'stage_a_results.json').read_text())
    d['stage_a'] = stage['switch']['30']
    d['stage_a_sessions'] = stage_a_sessions()

    newton = [s for s in ALL_SESSIONS if s in NEWTON_EFFECT]
    for name, root in (('random', RES / 'grid_v12'), ('switch', RES / 'grid_v12s')):
        raw = load_raw(root)
        if raw is None:
            continue
        d[name] = {
            'all': grid_stats(raw, ALL_SESSIONS, model_on_test=(name == 'switch')),
            'newton': grid_stats(raw, newton, model_on_test=(name == 'switch')),
            'spatial': grid_stats(raw, ALL_SESSIONS, model_on_test=False, key='spatial'),
        }
    raw = load_raw(RES / 'grid_v12s')
    if raw is not None:
        ll = lambda v, sd: np.nanmean(np.concatenate([raw[f'{v}/{s}_seed{sd}']['ll'] for s in ALL_SESSIONS]))
        seeds = sorted({k.rsplit('_seed', 1)[1] for k in raw}, key=int)
        d['switch_ll_diff'] = [1000 * (ll('shared_free', sd) - ll('free_free', sd)) for sd in seeds]
    d['recovery'] = recovery_stats(RES / 'recovery_v12')
    d['recovery_drift'] = recovery_stats(RES / 'recovery_v12_drift')
    cal = RES / 'recovery_v12_drift' / 'drift_calibration.json'
    d['calibration'] = json.loads(cal.read_text()) if cal.exists() else None
    return d


def paired_diff(values: dict, a: str, b: str) -> list:
    return [x - y for x, y in zip(values.get(a, []), values.get(b, []))]


# ---------------------------------------------------------------- figures

def fig_population(d) -> str:
    rows = d['stage_a_sessions']
    p = Plot(848, 250, (-0.5, len(rows) - 0.5), (-0.4, 1.0), ml=40, mb=48, mt=24)
    for t in (-0.25, 0, 0.25, 0.5, 0.75, 1.0):
        p.hline(t, cls='zero' if t == 0 else 'grid')
    for k, r in enumerate(rows):
        for val, pv, fam, off, name in ((r['reward'], r['p_reward'], 'unit', -0.12, 'reward map'),
                                        (r['block'], r['p_block'], 'none', 0.12, 'probability block')):
            p.dot(k + off, val, fam, f"{label(r['session'])}, {name}: {val:+.2f} sp/s (p = {pv:g}, {r['n']} neurons)",
                  r=5.5, hollow=pv >= 0.05)
        p.add(f'<text class="tick" x="{p.X(k):.1f}" y="{p.h - 30}" text-anchor="middle">{esc(label(r["session"]))}</text>'
              f'<text class="tick" x="{p.X(k):.1f}" y="{p.h - 17}" text-anchor="middle">{r["n"]} neurons</text>')
    p.yaxis([-0.25, 0, 0.25, 0.5, 0.75, 1.0], '')
    p.add('<text class="ann" x="40" y="12">mean pre-target effect over switches (sp/s): '
          '<tspan class="t-unit">● reward map</tspan><tspan dx="12">● probability block</tspan>'
          '<tspan dx="12">filled: p &lt; 0.05</tspan></text>')
    tab = table(['Session', 'Neurons', 'Reward map (sp/s)', 'p', 'Probability block (sp/s)', 'p'],
                [[label(r['session']), r['n'], fmt(r['reward']), r['p_reward'], fmt(r['block']), r['p_block']] for r in rows])
    return p.svg('Population context effect per session') + tab


def fig_grid(d, name: str, title: str, ylab: str) -> str:
    vals, ceil = d[name]['all']
    order = [v for v in GRID_ORDER if v in vals]
    svg = strip_by_variant(vals, order, mean(ceil), title, ylab, yr=(-0.4, 0.8), h=250)
    newton, nceil = d[name]['newton']
    rows = [[R.VARIANTS[v][0], ' '.join(fmt(x) for x in vals[v]), fmt(mean(vals[v])), fmt(mean(newton.get(v, [])))]
            for v in order]
    rows.append(['ceiling', ' '.join(fmt(x) for x in ceil), fmt(mean(ceil)), fmt(mean(nceil))])
    return svg + table(['Variant', 'r per restart (7 sessions)', 'Mean', 'Mean, N 08_13 + 08_14'], rows)


def fig_differences(d) -> str:
    """Paired differences on real data: aligned vs shifted, context vs none, shared vs per-unit."""
    vals, _ = d['switch']['all']
    items = [('Per-unit → all minus its shifted control', paired_diff(vals, 'free_free', 'free_free_qshift'), 'unit'),
             ('Shared → all minus its shifted control', paired_diff(vals, 'shared_free', 'shared_free_qshift'), 'shared'),
             ('Per-unit → all minus no context', paired_diff(vals, 'free_free', 'none_free'), 'unit'),
             ('Shared → all minus no context', paired_diff(vals, 'shared_free', 'none_free'), 'shared'),
             ('Shared → all minus per-unit → all', paired_diff(vals, 'shared_free', 'free_free'), 'none')]
    return diff_plot(items, 'difference in held-out r (same split, per restart)', (-0.4, 0.4)), items


def diff_plot(items, xlab, xr) -> str:
    p = Plot(848, 50 + 34 * len(items), xr, (len(items) - 0.5, -0.5), ml=300, mt=16, mb=34)
    for t in np.arange(xr[0], xr[1] + 1e-9, 0.1):
        p.vline(round(t, 2), cls='zero' if abs(t) < 1e-9 else 'grid')
    for k, (lab, vals, fam) in enumerate(items):
        p.add(f'<text class="tick" x="{p.ml - 10}" y="{p.Y(k) + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
        for j, x in enumerate(vals):
            if not np.isnan(x):
                p.dot(x, k, fam, f'{lab}, restart {j + 1}: {x:+.2f}', r=3.5)
        m = mean(vals)
        if not np.isnan(m):
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            p.add(f'<line class="meanbar {fam}" x1="{p.X(m):.1f}" x2="{p.X(m):.1f}" y1="{p.Y(k - 0.3):.1f}" '
                  f'y2="{p.Y(k + 0.3):.1f}"><title>{esc(lab)}: mean {m:+.2f} ± {se:.2f} (SE)</title></line>')
    p.xaxis([round(t, 1) for t in np.arange(xr[0], xr[1] + 1e-9, 0.2)], xlab, fmt='{:+.1f}')
    return p.svg('Paired differences')


def fig_recovery(rec: dict, title_suffix: str) -> str:
    out, rows = [], []
    for t, (lab, fam) in TEACHERS.items():
        if t not in rec:
            out.append(f'<div class="pending"><p class="ann">{esc(lab)} teacher: <strong>still running</strong></p></div>')
            continue
        r = rec[t]
        svg = strip_by_variant(r['students'], STUDENTS, r['ceiling'], f'{lab} teacher{title_suffix}',
                               f'{lab} teacher', yr=(-0.4, 0.8), w=420, h=220,
                               ceiling_label=f'ceiling {abs(r["ceiling"]) if abs(r["ceiling"]) < 0.005 else r["ceiling"]:.2f}', mean_labels=False)
        out.append(f'<div class="panelwrap">{svg}</div>')
        rows.append([lab, f'{r["ceiling"]:.2f}'] + [fmt(mean(r['students'][v])) for v in STUDENTS])
    return ('<div class="grid2">' + ''.join(out) + '</div>'
            + table(['Teacher', 'Ceiling'] + [R.VARIANTS[v][0] for v in STUDENTS], rows))


def fig_calibrated(d) -> str:
    """Shared minus per-unit student, per teacher (both recovery sets) and on real data."""
    items = []
    vals, _ = d['switch']['all']
    items.append(('Real data, 7 sessions (ceiling {:.2f})'.format(mean(d['switch']['all'][1])),
                  paired_diff(vals, 'shared_free', 'free_free'), 'none'))
    for key, tag in (('recovery_drift', 'realistic'), ('recovery', 'idealized')):
        for t in ('shared', 'ext', 'local', 'null'):
            if t in d[key]:
                r = d[key][t]
                items.append((f'{TEACHERS[t][0]} teacher, {tag} (ceiling {r["ceiling"]:.2f})', r['diff'], TEACHERS[t][1]))
    return diff_plot(items, 'shared → all minus per-unit → all student, held-out r', (-0.5, 0.4))


def embed_png(path: Path, width: int = 1700) -> str:
    """Downscaled JPEG data URI of a figure (the page stays self-contained)."""
    if not path.exists():
        return '<div class="pending"><p class="ann">figure missing</p></div>'
    from PIL import Image
    img = Image.open(path).convert('RGB')
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=86)
    return (f'<img class="fig" alt="{esc(path.stem)}" src="data:image/jpeg;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}">')


# ---------------------------------------------------------------- page

EXTRA_CSS = """
img.fig { width:100%; height:auto; border-radius:4px; background:#fff; }
.wide { margin-left:-8px; margin-right:-8px; }
"""


def main():
    d = gather()
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    c = d['counts']
    n_e, n_i = sum(x['E'] for x in c.values()), sum(x['I'] for x in c.values())
    n_tr, n_sw = sum(x['trials'] for x in c.values()), sum(x['switches'] for x in c.values())
    sa = d['stage_a']['reward|pre|all']
    rnd, sw = d['random']['all'], d['switch']['all']
    g = lambda which, v: mean(which[0].get(v, []))
    diffs_svg, diffs = fig_differences(d)
    dm = {lab: (mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) for lab, v, _ in diffs}
    sh_free = dm['Shared → all minus per-unit → all']
    rd, rv = d['recovery_drift'], d['recovery']
    have_drift = all(t in rd for t in TEACHERS)
    cal = d['calibration']
    legend = ('<p class="legendrow">Colour code:<span class="sw" style="background:var(--none)"></span>no context or control'
              '<span class="sw" style="background:var(--shared)"></span>shared context'
              '<span class="sw" style="background:var(--unit)"></span>per-unit context.</p>')

    rec_text = recovery_text(rv, rd, have_drift)
    body = f"""
<h1>Reward-context inputs to SC on the corrected exports: what the RNN fits can and cannot tell apart</h1>
<div class="meta">Corrected exports (pipeline 1.2.0), Feynman 08_12–08_15 and Newton 08_13–08_15. Built
{esc(__import__('datetime').date.today())} from commit <code>{esc(commit)}</code> by <code>scripts/make_context_report.py</code>.
Prepared by Claude for James Herman. Supersedes the conclusions of <code>results/recovery_pilot_report.html</code>.</div>

<section class="abstract">
<p>Before the target appears, SC neurons fire slightly more when their preferred location currently gives the large
reward. We asked whether E–I network models need a separate reward-context input for each neuron to reproduce
which neurons show this effect, or whether one shared reward signal, routed by the local circuit, would do.</p>
<p>{summary_text(d, g, sh_free, have_drift)}</p>
</section>
{legend}

<h2>1. Data</h2>
<p><span class="tag">S</span>{n_e + n_i} neurons ({n_e} classic SC neurons fit by E units, {n_i} putative interneurons
fit by I units) in 7 sessions; {n_tr} successful trials whose event times were verified. Each neuron is analysed only
within its stable recording period ({min(x['stable'] for x in c.values()):.0%}–{max(x['stable'] for x in c.values()):.0%}
of neuron-trials per session). The reward map switched every 48 successful trials, {n_sw} switches in all.</p>
{table(['Session', 'Trials', 'E', 'I', 'Reward switches', 'Stable neuron-trials'],
       [[label(s), x['trials'], x['E'], x['I'], x['switches'], f"{x['stable']:.1%}"] for s, x in c.items()])}

<h2>2. The population effect is real but small, and comes from two sessions</h2>
<p><span class="tag">F</span>Pre-target firing rose by {sa['mean_diff_sp_s']:+.2f} sp/s on average when a neuron's preferred
location became the large-reward location (30 trials after vs before each switch; p = {sa['p_switches']:.1g} across
{sa['n_switches']} switches). The effect is significant in Newton 08_13 and 08_14 only (Fig. 1). Newton 08_14 has the
most slow drift of any session; the switch-triggered contrast is local, but conclusions resting on that session alone
should be treated with caution.</p>
<figure>{fig_population(d)}
<figcaption><strong>Figure 1. The reward-context effect is carried by Newton 08_13 and 08_14.</strong> Mean
switch-triggered pre-target effect per session for the reward map and the probability block; filled markers p &lt; 0.05.</figcaption></figure>

<h2>3. Models</h2>
<p><span class="tag">S</span>Each model is an E–I rate network with one unit per recorded neuron, Dale's law and local-only
inhibition. It receives the task events, eye position, three slow-drift inputs, and (except the no-context variant) two
context inputs: the reward map and the probability block. Variants differ only in how the context reaches the units
(Fig. 2). Each variant was fit 6 times from different random starts ("restarts").</p>
<figure>{fig_models()}
<figcaption><strong>Figure 2. The variants differ only in how the context inputs reach the network.</strong></figcaption></figure>

<h2>4. An interleaved test set overstates what the models capture</h2>
<p><span class="tag">F</span>With test trials scattered through the session, per-unit models reproduced the neuron-by-neuron
pattern well (r = {g(rnd, 'free_free'):.2f} for per-unit → all; ceiling {mean(rnd[1]):.2f}; Fig. 3). But a control whose
context input was shifted a quarter cycle out of register with the real context reached {g(rnd, 'free_free_qshift'):.2f}.
That shifted input carries no reward information at the switches, so most of the score came from something else:
per-unit inputs let each unit follow its neuron's slow fluctuations on the training trials, and those fluctuations are
shared with the neighbouring test trials. Shared-context models ({g(rnd, 'shared_free'):.2f}) scored about as high as
the shifted control.</p>
<figure>{fig_grid(d, 'random', 'Interleaved split', 'held-out r with the recorded reward-context effect (interleaved test trials)')}
<figcaption><strong>Figure 3. With interleaved test trials, a context input shifted out of register scores nearly as
well as the real one.</strong> One dot per restart, bar = mean, green line = reliability ceiling of the recorded effect.</figcaption></figure>

<h2>5. Holding out whole switches: the pattern is weak, and shared and per-unit context are indistinguishable</h2>
<p><span class="tag">S</span>We refit every variant holding out whole windows of ±30 trials around 2 random reward switches
for testing (and 2 for checkpoint selection), different switches for each restart. A test switch then shares no slow
fluctuations with training trials, and any effect a model predicts there must come from its context input.</p>
<p><span class="tag">F</span>The ceiling dropped from {mean(rnd[1]):.2f} to {mean(sw[1]):.2f}: measured across switches, each
neuron's reward-context effect is only weakly reproducible. Against this ceiling, models with aligned context scored
{min(g(sw, v) for v in ('shared_free', 'sharedI_free', 'freeI_free', 'freeE_free', 'free_free')):.2f}–{max(g(sw, v) for v in ('shared_free', 'sharedI_free', 'freeI_free', 'freeE_free', 'free_free')):.2f},
the shifted controls {g(sw, 'free_free_qshift'):.2f} and {g(sw, 'shared_free_qshift'):.2f}, and no context
{g(sw, 'none_free'):.2f}. The no-context score is not zero, because which trials fall in the test windows already shapes
the measured effect. Paired over restarts (Fig. 5), aligned context beats its shifted control by about 0.1, within 1–2
standard errors, and shared minus per-unit is {sh_free[0]:+.2f} ± {sh_free[1]:.2f}.</p>
<figure>{fig_grid(d, 'switch', 'Switch-window split', 'held-out r with the recorded reward-context effect (held-out switches)')}
<figcaption><strong>Figure 4. On held-out switches, no variant clearly beats the controls.</strong> Model effects measured on
the same held-out trials as the recorded effect. Layout as Fig. 3.</figcaption></figure>
<figure>{diffs_svg}
<figcaption><strong>Figure 5. Paired differences between variants on the same held-out switches.</strong> Dots: restarts;
bars: means (hover for SE).</figcaption></figure>

<h2>6. Recovery test: can the procedure tell a shared from a per-unit mechanism?</h2>
<p><span class="tag">S</span>Teachers are networks already fit to each session, whose mechanism is therefore known: no
context, shared context (gain set so that its context-driven effects are as large as the per-unit teacher's), per-unit
context to all units, and per-unit context to I units only. Spikes were generated from each teacher on the real trials,
and students of every variant were fit to them exactly as to the real data, with the switch-window split.</p>
{rec_text}
<figure>{fig_recovery(rv, ', idealized')}
<figcaption><strong>Figure 6. Idealized teachers</strong> (no slow fluctuations beyond their own drift inputs; 3 restarts
per student; 7 sessions pooled). Panels: teachers; columns: students.</figcaption></figure>
<figure>{fig_recovery(rd, ', realistic') if rd else '<div class="pending"><p class="ann">Realistic teachers: <strong>still running</strong></p></div>'}
<figcaption><strong>Figure 7. Teachers with realistic slow fluctuations</strong> (each recorded neuron's own slow rate changes,
shifted out of register with context; strength set per teacher so the ceiling matches the recordings; 6 restarts per student).</figcaption></figure>
<figure>{fig_calibrated(d)}
<figcaption><strong>Figure 8. The key contrast: shared minus per-unit student, on real data and on each teacher.</strong>
With idealized teachers, negative values point to a per-unit mechanism; with realistic teachers the contrast is
negative for every mechanism. Dots: restarts; bars: means.</figcaption></figure>
{realistic_comparison(d)}

<h2>7. Fit quality</h2>
<p><span class="tag">F</span>The models fit trial-averaged responses well where the recordings are reliable (Fig. 9: median
r 0.60 against a median split-half reliability of 0.52 over 270 units; r and reliability correlate across units). Visual
selectivity for location is reproduced on held-out trials (Fig. 10: r 0.57, range 0.32–0.84 over restarts); identity,
salience and probability selectivity are not.</p>
<figure class="wide">{embed_png(RES / 'psth_match' / 'psth_best_worst_free_free.png')}
<figcaption><strong>Figure 9. Condition PSTHs, recorded vs model</strong> (per-unit → all, interleaved-split fits, best
restart per session): best and worst matches, and the reliable units the model fits worst.</figcaption></figure>
<figure class="wide">{embed_png(RES / 'r01_figure' / 'figure_recorded_vs_rnn_grid_v12.png')}
<figcaption><strong>Figure 10. Visual-epoch selectivity, recorded vs model</strong> (50–250 ms after target onset, held-out
trials, 6 restarts).</figcaption></figure>

<h2>8. Conclusions and open problems</h2>
{conclusions_text(d, g, sh_free, have_drift)}
"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Reward Context Report</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>{body}</main></body></html>')
    OUT.write_text(page, encoding='utf-8')
    print(f"Wrote {OUT} ({len(page) / 1024:.0f} KB)")


def summary_text(d, g, sh_free, have_drift) -> str:
    sw, rnd = d['switch']['all'], d['random']['all']
    s = (f"On interleaved test trials, per-unit models appeared to reproduce the pattern (r = {g(rnd, 'free_free'):.2f}), but "
         f"a control with context shifted out of register scored almost as well ({g(rnd, 'free_free_qshift'):.2f}): the "
         f"score mostly reflected slow fluctuations shared between training and test trials. Holding out whole reward "
         f"switches removes that leak. The neuron-by-neuron pattern then barely repeats from switch to switch (ceiling "
         f"{mean(sw[1]):.2f}), aligned context gains only about 0.1 over its controls, and shared and per-unit context fit "
         f"equally well ({sh_free[0]:+.2f} ± {sh_free[1]:.2f}). ")
    rv = d['recovery']
    if all(t in rv for t in TEACHERS):
        s += (f"With idealized teachers the procedure separates the mechanisms clearly (shared minus per-unit "
              f"{mean(rv['ext']['diff']):+.2f} and {mean(rv['local']['diff']):+.2f} for per-unit teachers, "
              f"{mean(rv['shared']['diff']):+.2f} for a shared one). ")
    if have_drift:
        rd = d['recovery_drift']
        s += (f"With teachers as noisy as the data, it no longer does: the contrast is negative for every teacher, "
              f"{mean(rd['ext']['diff']):+.2f} and {mean(rd['local']['diff']):+.2f} for per-unit teachers but also "
              f"{mean(rd['shared']['diff']):+.2f} for the shared teacher and {mean(rd['null']['diff']):+.2f} with no context "
              f"at all. Held-out likelihood separates the two kinds of teacher only weakly. "
              "With the present number of reward switches, these fits cannot decide whether SC reward-context input is "
              "neuron-specific or a shared signal routed by the local circuit; the data favour neither.")
    else:
        s += ("Whether it still does with teachers as noisy as the data is running now. The data therefore do not show "
              "that SC reward-context input must be neuron-specific.")
    return s


def recovery_text(rv, rd, have_drift) -> str:
    t = []
    if all(k in rv for k in TEACHERS):
        t.append(f"<p><span class=\"tag\">F</span>With idealized teachers the procedure works. No student found a reward "
                 f"pattern in the no-context teacher (ceiling {rv['null']['ceiling']:.2f}). Each context teacher was recovered "
                 f"best by its own kind of student, and shared minus per-unit students separated the mechanisms in every "
                 f"restart: {mean(rv['ext']['diff']):+.2f} and {mean(rv['local']['diff']):+.2f} for the per-unit teachers, "
                 f"{mean(rv['shared']['diff']):+.2f} for the shared teacher (Figs. 6, 8). A shared signal also produced a "
                 f"neuron-specific pattern that per-unit students fit well ({mean(rv['shared']['students']['free_free']):.2f}), "
                 f"so a good per-unit fit is not by itself evidence for per-unit input.</p>")
        t.append(f"<p><span class=\"tag\">F</span>But these teachers are easier than the data. Their ceilings "
                 f"({min(rv[k]['ceiling'] for k in ('ext', 'local', 'shared')):.2f}–{max(rv[k]['ceiling'] for k in ('ext', 'local', 'shared')):.2f}) "
                 f"are far above the recordings', because a teacher's context effect is the same at every switch, while "
                 f"recorded neurons also fluctuate slowly for other reasons. We therefore multiplied each teacher's rates "
                 f"by its recorded neuron's own slow rate fluctuation (smoothed over about 15 trials, mean preserved), shifted "
                 f"by whole reward cycles plus a quarter cycle so that it carries no reward-context signal, with a strength "
                 f"per teacher chosen so that its ceiling matches the recordings.</p>")
    if have_drift:
        se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
        pm = lambda v, f='+.2f': f"{mean(v):{f}} ± {se(v):{f[1:]}}"
        t.append(f"<p><span class=\"tag\">F</span>With these realistic teachers (ceilings "
                 f"{min(rd[k]['ceiling'] for k in ('ext', 'local', 'shared')):.2f}–{max(rd[k]['ceiling'] for k in ('ext', 'local', 'shared')):.2f}), "
                 f"the contrast loses its power (Figs. 7, 8). Shared minus per-unit was {pm(rd['ext']['diff'])} and "
                 f"{pm(rd['local']['diff'])} for the per-unit teachers, but {pm(rd['shared']['diff'])} for the shared "
                 f"teacher and {pm(rd['null']['diff'])} for the no-context teacher. Per-unit students now score higher "
                 f"whatever the mechanism: their extra freedom lets them follow fluctuations at the held-out switches "
                 f"that are unrelated to context. Even with no context in the teacher, per-unit students scored "
                 f"{mean(rd['null']['students']['free_free']):.2f} against a ceiling of {rd['null']['ceiling']:.2f}, so "
                 f"scores on held-out switches are biased upwards at this noise level.</p>")
    else:
        t.append("<p><span class=\"tag\">S</span>The realistic-teacher runs (840 student fits) are running on the cluster; "
                 "Fig. 7 and the realistic rows of Fig. 8 fill in when they finish.</p>")
    return ''.join(t)


def realistic_comparison(d) -> str:
    """Real data against the realistic teachers, on both contrasts."""
    rd = d['recovery_drift']
    if not all(t in rd for t in TEACHERS):
        return ''
    se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
    real_r = paired_diff(d['switch']['all'][0], 'shared_free', 'free_free')
    real_ll = d.get('switch_ll_diff', [])
    rows = [['Real data', f"{mean(real_r):+.2f} ± {se(real_r):.2f}",
             f"{mean(real_ll):+.1f} ± {se(real_ll):.1f}" if real_ll else '–']]
    for t in ('shared', 'ext', 'local', 'null'):
        rows.append([f'{TEACHERS[t][0]} teacher', f"{mean(rd[t]['diff']):+.2f} ± {se(rd[t]['diff']):.2f}",
                     f"{mean(rd[t]['ll_diff']):+.1f} ± {se(rd[t]['ll_diff']):.1f}"])
    return (table(['', 'Shared − per-unit, held-out r', 'Shared − per-unit, held-out LL (×10⁻³)'], rows)
            .replace('<details>', '<details open>').replace('<summary>Table view</summary>', '<summary>Real data vs realistic teachers</summary>')
            + f"<p><span class=\"tag\">F</span>The real-data r contrast ({mean(real_r):+.2f}) lies above every realistic "
              f"teacher, including the no-context one, so it does not match any of them well; the earlier argument that "
              f"it sits on the shared prediction, made by scaling the idealized teachers' values by ceiling, does not "
              f"survive the realistic test. Held-out likelihood behaves more as expected (positive for the shared teacher, "
              f"negative for the per-unit → all teacher, near zero for the per-unit → I and no-context teachers), but its "
              f"restart-to-restart spread is large, and the real-data value ({mean(real_ll):+.1f}) is within about one "
              f"standard error of the shared, per-unit → I and no-context teachers, and about two from the "
              f"per-unit → all teacher. Values are mean ± SE over 6 restarts.</p>")


def conclusions_text(d, g, sh_free, have_drift) -> str:
    items = [
        "<strong>Reward context modulates SC pre-target activity weakly</strong> (+0.28 sp/s), mainly in two Newton "
        "sessions.",
        "<strong>The neuron-by-neuron pattern is not reliably measurable across switches</strong> with these data "
        f"(ceiling {mean(d['switch']['all'][1]):.2f}). Earlier, higher reliability came from slow fluctuations, not context.",
        "<strong>Shared and per-unit context fit the real data equally well</strong>; the earlier conclusion that only "
        "per-unit context reproduces the pattern is withdrawn.",
    ]
    if have_drift:
        items.append("<strong>With data-like noise, the fitting procedure cannot tell shared from per-unit context.</strong> "
                     "On teachers whose reliability matches the recordings, per-unit students win the held-out-r contrast "
                     "whatever the teacher's mechanism, and held-out likelihood separates mechanisms only weakly. The "
                     "earlier reading that the real data match the shared prediction is withdrawn: the data favour neither.")
        items.append("<strong>Scores on held-out switches are biased upwards</strong> at this noise level (per-unit students "
                     "score about 0.2 on a teacher with no context), so they should be compared against matched "
                     "synthetic controls, not against zero or the ceiling.")
    gaps = [
        "<strong>More switches per neuron</strong> would be needed to measure neuron-specific context effects: the ceiling "
        "rises with the number of switches averaged.",
        "<strong>The realistic teachers still differ from the data.</strong> On held-out switches, per-unit students beat "
        "no-context students by 0.13–0.31 on every teacher, but by only about 0.02 on the real data, and the real "
        "shared − per-unit contrast lies above every teacher's. Teacher noise is a per-neuron gain on the neuron's own slow "
        "rate changes; real fluctuations may be shared across neurons or structured differently.",
        "<strong>E vs I as the landing site</strong> of context input is only weakly identifiable even with idealized teachers.",
        "<strong>Newton 08_14</strong> has strong slow drift in many neurons (they are real neurons, not split clusters); "
        "possible cluster merges in the pipeline could change its inputs.",
        "<strong>Not reproduced by any model:</strong> the visual-response probability effect (−0.80 sp/s), and high-rate "
        "transients (training clips rates at 120–160 sp/s).",
    ]
    return ('<ul>' + ''.join(f'<li>{x}</li>' for x in items) + '</ul>'
            + ''.join(f'<div class="gap"><span class="tag">Open</span>{x}</div>' for x in gaps))


if __name__ == '__main__':
    main()
