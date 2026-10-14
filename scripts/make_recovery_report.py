#!/usr/bin/env python3
"""
Build results/recovery_pilot_report.html: a self-contained report on the recovery pilot
(results/recovery_pilot) and the verified-trial context grid it calibrates
(results/verified_grid). Figures are inline SVG generated from the evaluation files, so
the report can be rebuilt whenever those change.

Usage:
    python scripts/make_recovery_report.py
"""

import html
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

import analyze_context_effects as A
import evaluate_context_models as E
from src.analysis.task_context import get_switches, location_rewarded
from src.sessions import load_export

ROOT = Path(__file__).parent.parent
GRID = ROOT / 'results' / 'verified_grid'
PILOT = ROOT / 'results' / 'recovery_pilot'
OUT = ROOT / 'results' / 'recovery_pilot_report.html'

# Context-routing variants: label, family (colour), short description
VARIANTS = {
    'none_free': ('No context', 'none', 'no context input'),
    'shared_free': ('Shared → all', 'shared', 'one weight per population (E, I)'),
    'sharedI_free': ('Shared → I', 'shared', 'one weight, I units only'),
    'sharedE_free': ('Shared → E', 'shared', 'one weight, E units only'),
    'freeI_free': ('Per-unit → I', 'unit', 'own weight for each I unit'),
    'freeE_free': ('Per-unit → E', 'unit', 'own weight for each E unit'),
    'free_free': ('Per-unit → all', 'unit', 'own weight for every unit'),
    # Control: per-unit context, each channel shifted by a quarter of its period (uncorrelated with the truth)
    'freeI_free_qshift': ('Shifted per-unit → I', 'none', 'control: context shifted out of register'),
    'free_free_qshift': ('Shifted per-unit → all', 'none', 'control: context shifted out of register'),
}
STUDENTS = ['none_free', 'shared_free', 'sharedI_free', 'freeI_free', 'free_free']
TEACHERS = {  # dataset: (label, family, gain)
    'null': ('No-context teacher', 'none', None),
    'shared_g1': ('Shared teacher, g = 1', 'shared', 1),
    'shared_g3': ('Shared teacher, g = 3', 'shared', 3),
    'ext_g1': ('Per-unit (all) teacher, g = 1', 'unit', 1),
    'ext_g3': ('Per-unit (all) teacher, g = 3', 'unit', 3),
    'local_g1': ('Per-unit (I only) teacher, g = 1', 'unit', 1),
    'local_g3': ('Per-unit (I only) teacher, g = 3', 'unit', 3),
}
FAMILY_LABEL = {'none': 'no context', 'shared': 'shared context', 'unit': 'per-unit context'}


# ---------------------------------------------------------------- data

def load_summary(root: Path):
    path = root / 'evaluation_summary.json'
    return json.loads(path.read_text()) if path.exists() else None


def per_restart(summary: dict, variant: str, key: str) -> list:
    if summary is None or variant not in summary or 'per_seed' not in summary[variant]:
        return []
    return [e.get(key, np.nan) for e in summary[variant]['per_seed']]


def gather() -> dict:
    d = {}
    grid = load_summary(GRID)
    d['grid'] = {v: {'r': per_restart(grid, v, 'reward_pre_r_test'),
                     'spatial': per_restart(grid, v, 'spatial_r'),
                     'block': per_restart(grid, v, 'block_pre_r_test'),
                     'dll': per_restart(grid, v, 'dll')}
                 for v in VARIANTS if grid and v in grid}
    d['grid_ceiling'] = float(np.nanmean(per_restart(grid, 'none_free', 'reward_pre_ceiling_test')))
    d['grid_spatial_ceiling'] = float(np.nanmean(per_restart(grid, 'none_free', 'spatial_ceiling_test')))
    d['grid_block_ceiling'] = float(np.nanmean(per_restart(grid, 'none_free', 'block_pre_ceiling_test')))

    d['pilot'] = {}
    for ds in TEACHERS:
        s = load_summary(PILOT / ds)
        if s is None:
            continue
        d['pilot'][ds] = {'ceiling': float(np.nanmean(per_restart(s, 'none_free', 'reward_pre_ceiling_test'))),
                          'recorded_mean': float(np.nanmean(per_restart(s, 'none_free', 'reward_pre_recorded'))),
                          'students': {v: per_restart(s, v, 'reward_pre_r_test') for v in STUDENTS}}

    sessions = {s: A.load_session(s, 'verified') for s in A.SESSIONS}

    # Per-neuron scatter: best restart (selection LL) per session, pooled over sessions
    raw = np.load(GRID / 'evaluation_raw.npy', allow_pickle=True).item()
    d['scatter'] = {}
    for v in ('free_free', 'shared_free'):
        model, rec, is_e = [], [], []
        for s in A.SESSIONS:
            runs = {k: r for k, r in raw.items() if k.startswith(f'{v}/{s}_seed')}
            best = max(runs.values(), key=lambda r: r['select_ll'])
            model.append(best['effects']['reward_pre'])
            rec.append(best['recorded_split']['test']['reward_pre'])
            is_e.append(sessions[s]['is_e'])
        d['scatter'][v] = (np.concatenate(model), np.concatenate(rec), np.concatenate(is_e))

    # Recorded data: effects, reliability, an example neuron
    eff = {s: E.effects(sess)['reward_pre'] for s, sess in sessions.items()}
    d['effects'] = np.concatenate([eff[s] for s in A.SESSIONS])
    d['effects_is_e'] = np.concatenate([sessions[s]['is_e'] for s in A.SESSIONS])
    any_run = {s: next(r for k, r in raw.items() if k.startswith(f'none_free/{s}_seed1')) for s in A.SESSIONS}
    d['split_rest'] = np.concatenate([any_run[s]['recorded_split']['rest']['reward_pre'] for s in A.SESSIONS])
    d['split_test'] = np.concatenate([any_run[s]['recorded_split']['test']['reward_pre'] for s in A.SESSIONS])
    d['counts'] = {}
    for s in A.SESSIONS:
        n_all = load_export(s, 'all')['firing_rates'].shape[-1]
        nt = np.asarray(sessions[s]['data']['neuron_type']).ravel()
        d['counts'][s] = {'exported': int(n_all), 'verified': len(sessions[s]['location']),
                          'E': int((nt == 1).sum()), 'I': int((nt == 2).sum()),
                          'reward_switches': len(get_switches(sessions[s]['context']['reward_map'].astype(float)))}
    s = sessions['08_15']
    d['timeline'] = {'reward_map': s['context']['reward_map'].astype(int).tolist(),
                     'block': np.nan_to_num(s['context']['block'], nan=0).astype(int).tolist()}
    i, t = 19, 61  # 08_15 neuron 19 (E, prefers location 1); reward-map switch at trial 61
    lo, hi = t - 30, t + 30
    gains = location_rewarded(s['context']['reward_map'], int(s['preferred']['for_factors'][i]))
    d['example'] = {'trials': list(range(lo, hi)), 'rate': s['pre'][lo:hi, i].tolist(),
                    'gain': gains[lo:hi].tolist(), 'switch': t,
                    'before': float(np.nanmean(s['pre'][lo:t, i])), 'after': float(np.nanmean(s['pre'][t:hi, i])),
                    'neuron_effect': float(eff['08_15'][i])}
    return d


# ---------------------------------------------------------------- SVG helpers

def esc(x) -> str:
    return html.escape(str(x))


class Plot:
    """Minimal SVG plot: linear axes, recessive grid, marks with native tooltips."""

    def __init__(self, w, h, xr, yr, ml=48, mr=12, mt=12, mb=34):
        self.w, self.h, self.xr, self.yr = w, h, xr, yr
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.items = []

    def X(self, x):
        return self.ml + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.w - self.ml - self.mr)

    def Y(self, y):
        return self.mt + (self.yr[1] - y) / (self.yr[1] - self.yr[0]) * (self.h - self.mt - self.mb)

    def add(self, s):
        self.items.append(s)

    def hline(self, y, cls='grid', label=None, anchor='end'):
        self.add(f'<line class="{cls}" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.Y(y):.1f}" y2="{self.Y(y):.1f}"/>')
        if label:
            x = self.w - self.mr - 2 if anchor == 'end' else self.ml + 4
            dy = 12 if self.Y(y) - self.mt < 14 else -4  # below the line when it sits at the top
            self.add(f'<text class="ref" x="{x}" y="{self.Y(y) + dy:.1f}" text-anchor="{anchor}">{esc(label)}</text>')

    def vline(self, x, cls='grid', label=None):
        self.add(f'<line class="{cls}" y1="{self.mt}" y2="{self.h - self.mb}" x1="{self.X(x):.1f}" x2="{self.X(x):.1f}"/>')
        if label:
            self.add(f'<text class="ref" x="{self.X(x) + 4:.1f}" y="{self.mt + 10}">{esc(label)}</text>')

    def yaxis(self, ticks, label):
        for t in ticks:
            self.add(f'<text class="tick" x="{self.ml - 6}" y="{self.Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
        self.add(f'<text class="axlabel" x="{self.ml}" y="{self.mt - 2 if self.mt > 14 else 10}" '
                 f'text-anchor="start">{esc(label)}</text>' if self.mt > 14 else '')
        self.ylabel = label

    def xaxis(self, ticks, label, fmt='{:g}'):
        y0 = self.h - self.mb
        self.add(f'<line class="axis" x1="{self.ml}" x2="{self.w - self.mr}" y1="{y0}" y2="{y0}"/>')
        for t in ticks:
            self.add(f'<text class="tick" x="{self.X(t):.1f}" y="{y0 + 14}" text-anchor="middle">{fmt.format(t)}</text>')
        self.add(f'<text class="axlabel" x="{(self.ml + self.w - self.mr) / 2:.1f}" y="{self.h - 4}" '
                 f'text-anchor="middle">{esc(label)}</text>')

    def dot(self, x, y, cls, tip, r=4.5, hollow=False):
        fill = 'none' if hollow else ''
        self.add(f'<circle class="mk {cls}{" hollow" if hollow else ""}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" '
                 f'r="{r}"><title>{esc(tip)}</title></circle>')

    def svg(self, title, ylabel_rot=None):
        rot = ''
        if ylabel_rot:
            cy = (self.mt + self.h - self.mb) / 2
            rot = (f'<text class="axlabel" transform="translate(12,{cy:.1f}) rotate(-90)" '
                   f'text-anchor="middle">{esc(ylabel_rot)}</text>')
        return (f'<svg width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(title)}">'
                + rot + ''.join(self.items) + '</svg>')


def table(headers, rows) -> str:
    th = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    trs = ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in r) + '</tr>' for r in rows)
    return f'<details><summary>Table view</summary><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></details>'


def fmt(x, n=2):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:+.{n}f}'


def mean(xs):
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else np.nan


# ---------------------------------------------------------------- figures

def fig_task(d) -> str:
    """Task layout, block structure, trial epochs, trial selection."""
    # (a) locations and the two context variables
    a = ['<svg width="320" height="220" viewBox="0 0 320 220" role="img" aria-label="Target locations and context">']
    pos = {1: (90, 60), 2: (90, 150), 3: (230, 150), 4: (230, 60)}
    a.append('<text class="panel" x="4" y="14">a</text>')
    a.append('<circle cx="160" cy="105" r="4" class="fix"/><text class="ann" x="160" y="124" text-anchor="middle">fixation</text>')
    for loc, (x, y) in pos.items():
        a.append(f'<circle cx="{x}" cy="{y}" r="15" class="loc"/><text class="locnum" x="{x}" y="{y + 5}" text-anchor="middle">{loc}</text>')
    a.append('<text class="ann" x="160" y="22" text-anchor="middle">4 target locations, one target per trial</text>')
    a.append('<text class="ann" x="160" y="194" text-anchor="middle">Reward map: 1 + 2 or 3 + 4 give the large reward</text>')
    a.append('<text class="ann" x="160" y="210" text-anchor="middle">Probability block: 1 or 3 is the frequent target</text>')
    a.append('</svg>')

    # (d) trial selection
    p = Plot(420, 150, (0, 1100), (0, 3), ml=56, mr=80, mb=34, mt=10)
    for k, s in enumerate(A.SESSIONS):
        cn = d['counts'][s]
        y = 2.5 - k
        top, hgt = p.Y(y + 0.3), p.Y(y - 0.3) - p.Y(y + 0.3)
        p.add(f'<rect class="bar excluded" x="{p.X(0):.1f}" y="{top:.1f}" width="{p.X(cn["exported"]) - p.X(0):.1f}" '
              f'height="{hgt:.1f}" rx="2"><title>{s}: {cn["exported"]} exported trials</title></rect>')
        p.add(f'<rect class="bar kept" x="{p.X(0):.1f}" y="{top:.1f}" width="{p.X(cn["verified"]) - p.X(0):.1f}" '
              f'height="{hgt:.1f}" rx="2"><title>{s}: {cn["verified"]} verified successful trials</title></rect>')
        p.add(f'<text class="tick" x="{p.ml - 6}" y="{p.Y(y) + 4:.1f}" text-anchor="end">{s}</text>')
        p.add(f'<text class="ann" x="{p.X(cn["exported"]) + 5:.1f}" y="{p.Y(y) + 4:.1f}">{cn["verified"]} of {cn["exported"]}</text>')
    p.xaxis([0, 500, 1000], 'trials: kept (solid) of exported (faint)')
    dsvg = '<div class="panelwrap"><span class="panel-l">d</span>' + p.svg('Trial selection') + '</div>'

    # (b) session timeline, 08_15
    rm, bl = np.array(d['timeline']['reward_map']), np.array(d['timeline']['block'])
    n = len(rm)
    w, h, ml, mr = 848, 108, 130, 10
    X = lambda t: ml + t / n * (w - ml - mr)
    b = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Context over one session">',
         '<text class="panel" x="4" y="14">b</text>']
    for row, (name, arr, labels) in enumerate([('Reward map', rm, {1: 'A (1 + 2 large)', 0: 'B (3 + 4 large)'}),
                                                ('Probability block', bl, {1: 'location 1 frequent', 2: 'location 3 frequent'})]):
        y = 18 + row * 30
        b.append(f'<text class="ann" x="{ml - 8}" y="{y + 13}" text-anchor="end">{name}</text>')
        start = 0
        for t in range(1, n + 1):
            if t == n or arr[t] != arr[start]:
                cls = 'segA' if arr[start] == 1 else 'segB'
                b.append(f'<rect class="{cls}" x="{X(start):.1f}" y="{y}" width="{max(X(t) - X(start) - 1.5, 0.5):.1f}" '
                         f'height="18"><title>{name}: {labels.get(int(arr[start]), "unknown")}, trials {start}–{t - 1}</title></rect>')
                start = t
    b.append(f'<line class="axis" x1="{ml}" x2="{w - mr}" y1="72" y2="72"/>')
    for t in range(0, n, 100):
        b.append(f'<text class="tick" x="{X(t):.1f}" y="86" text-anchor="middle">{t}</text>')
    b.append(f'<text class="axlabel" x="{(ml + w - mr) / 2:.1f}" y="104" text-anchor="middle">trial (session 08_15, {n} verified trials)</text>')
    b.append('</svg>')

    # (c) trial timeline
    w, h = 848, 124
    T = lambda ms: 60 + (ms + 750) / 1700 * (w - 90)
    c = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Trial timeline">',
         '<text class="panel" x="4" y="14">c</text>']
    c.append(f'<rect class="win pre" x="{T(-300):.1f}" y="30" width="{T(0) - T(-300):.1f}" height="44"/>')
    c.append(f'<text class="winlabel" x="{(T(-300) + T(0)) / 2:.1f}" y="24" text-anchor="middle">pre-target window, −300 to 0 ms: context effect measured here</text>')
    c.append(f'<rect class="win vis" x="{T(50):.1f}" y="30" width="{T(250) - T(50):.1f}" height="44"/>')
    c.append(f'<text class="winlabel" x="{(T(50) + T(250)) / 2:.1f}" y="88" text-anchor="middle">visual window, 50–250 ms</text>')
    c.append(f'<line class="axis" x1="{T(-750):.1f}" x2="{T(950):.1f}" y1="74" y2="74"/>')
    for ms, lab in [(-650, 'fixation point on'), (0, 'target on (0 ms)'), (650, 'go → saccade → reward')]:
        c.append(f'<line class="evt" x1="{T(ms):.1f}" x2="{T(ms):.1f}" y1="36" y2="78"/>'
                 f'<text class="ann" x="{T(ms):.1f}" y="108" text-anchor="middle">{lab}</text>')
    c.append(f'<text class="ref" x="{T(325):.1f}" y="120" text-anchor="middle">event times vary from trial to trial; windows are aligned to each trial\'s target onset</text>')
    c.append('</svg>')

    rows = [[s, c['exported'], c['verified'], c['E'], c['I'], c['reward_switches']] for s, c in d['counts'].items()]
    return (f'<div class="grid2">{"".join(a)}{dsvg}</div>{"".join(b)}{"".join(c)}'
            + table(['Session', 'Exported trials', 'Verified successful', 'E neurons', 'I neurons', 'Reward-map switches'], rows))


def fig_measure(d):
    ex = d['example']
    # (a) example neuron around one switch
    ymax = float(np.ceil(np.nanmax(ex['rate']) / 20) * 20)
    p = Plot(275, 220, (ex['trials'][0] - 1, ex['trials'][-1] + 1), (0, ymax), ml=36, mt=24)
    for y in np.arange(0, ymax + 1, 20):
        p.hline(y, cls='zero' if y == 0 else 'grid')
    p.vline(ex['switch'] - 0.5, cls='refline')
    p.add(f'<text class="ref" x="{p.X(ex["switch"]) + 3:.1f}" y="{p.mt + 9}">switch</text>')
    for t, r, g in zip(ex['trials'], ex['rate'], ex['gain']):
        if not np.isnan(r):
            p.dot(t, r, 'unit' if g else 'none', f'trial {t}: {r:.1f} sp/s; preferred location {"large" if g else "small"} reward', r=3)
    sw = ex['switch']
    p.add(f'<line class="meanline" x1="{p.X(ex["trials"][0]):.1f}" x2="{p.X(sw - 0.5):.1f}" y1="{p.Y(ex["before"]):.1f}" y2="{p.Y(ex["before"]):.1f}"/>')
    p.add(f'<line class="meanline" x1="{p.X(sw - 0.5):.1f}" x2="{p.X(ex["trials"][-1]):.1f}" y1="{p.Y(ex["after"]):.1f}" y2="{p.Y(ex["after"]):.1f}"/>')
    p.yaxis(list(np.arange(0, ymax + 1, 20)), '')
    p.add('<text class="ann" x="36" y="12">pre-target rate (sp/s)</text>')
    p.xaxis([ex['trials'][0], sw, ex['trials'][-1]], 'trial')
    a = '<div class="panelwrap"><span class="panel-l">a</span>' + p.svg('Example neuron') + '</div>'

    # (b) distribution of per-neuron effects
    eff, is_e = d['effects'], d['effects_is_e']
    lo, hi = np.floor(np.nanmin(eff)) - 0.5, np.ceil(np.nanmax(eff)) + 0.5
    p = Plot(275, 220, (lo, hi), (0, 2), ml=26, mt=24)
    p.vline(0, cls='zero')
    rng = np.random.default_rng(0)
    for v, e in zip(eff, is_e):
        if not np.isnan(v):
            y = (1.4 if e else 0.6) + rng.uniform(-0.25, 0.25)
            p.dot(v, y, 'unit', f'{"E" if e else "I"} neuron: {v:+.2f} sp/s', r=3, hollow=not e)
    m = np.nanmean(eff)
    p.add(f'<line class="meanline" x1="{p.X(m):.1f}" x2="{p.X(m):.1f}" y1="{p.Y(1.92):.1f}" y2="{p.Y(0.08):.1f}"/>')
    p.add(f'<text class="ref" x="{p.X(m) + 4:.1f}" y="{p.Y(1.92) + 8:.1f}">mean {m:+.2f}</text>')
    p.add(f'<text class="tick" x="20" y="{p.Y(1.4) + 4:.1f}" text-anchor="end">E</text><text class="tick" x="20" y="{p.Y(0.6) + 4:.1f}" text-anchor="end">I</text>')
    p.add('<text class="ann" x="26" y="12">one dot per neuron (n = 109)</text>')
    ticks = [t for t in (-5, 0, 5, 10) if lo <= t <= hi]
    p.xaxis(ticks, 'reward-context effect (sp/s)')
    b = '<div class="panelwrap"><span class="panel-l">b</span>' + p.svg('Effect distribution') + '</div>'

    # (c) reliability: test vs non-test trials
    x, y = d['split_rest'], d['split_test']
    ok = ~np.isnan(x) & ~np.isnan(y)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    lim = float(np.ceil(np.nanmax(np.abs(np.r_[x[ok], y[ok]])) / 5) * 5)
    p = Plot(275, 230, (-lim, lim), (-lim, lim), ml=36, mt=24)
    ticks = [t for t in (-10, 0, 10) if abs(t) <= lim]
    for t in ticks:
        p.hline(t, cls='zero' if t == 0 else 'grid')
    p.vline(0, cls='zero')
    p.add(f'<line class="diag" x1="{p.X(-lim):.1f}" y1="{p.Y(-lim):.1f}" x2="{p.X(lim):.1f}" y2="{p.Y(lim):.1f}"/>')
    for xi, yi in zip(x[ok], y[ok]):
        p.dot(xi, yi, 'none', f'other trials {xi:+.2f}, test trials {yi:+.2f} sp/s', r=3)
    p.yaxis(ticks, '')
    p.add(f'<text class="ann" x="36" y="12">test trials (sp/s); r = {r:.2f}</text>')
    p.xaxis(ticks, 'other 85% of trials (sp/s)')
    c = '<div class="panelwrap"><span class="panel-l">c</span>' + p.svg('Reliability') + '</div>'
    return f'<div class="grid3">{a}{b}{c}</div>', r


def fig_models() -> str:
    """Schematic of the network and the context-routing variants."""
    w, h = 848, 205
    s = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Model schematic">',
         '<text class="panel" x="4" y="14">a</text>',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>']
    for k, (lab, y) in enumerate([('task inputs (14): fixation, target location, stimulus, go, reward, eye position', 42),
                                   ('slow-drift inputs (3): smooth functions of trial number', 98),
                                   ('context inputs (2): reward map, probability block', 154)]):
        cls = 'ctxbox' if k == 2 else 'inbox'
        s.append(f'<rect class="{cls}" x="20" y="{y - 16}" width="430" height="30" rx="4"/>'
                 f'<text class="ann" x="30" y="{y + 3}">{lab}</text>')
        s.append(f'<line class="arrow" x1="452" y1="{y}" x2="508" y2="98" marker-end="url(#ah)"/>')
    s.append('<rect class="net" x="512" y="22" width="320" height="156" rx="8"/>')
    for k in range(8):
        s.append(f'<circle class="eunit" cx="{545 + (k % 4) * 26}" cy="{52 + (k // 4) * 28}" r="8"/>')
    for k in range(2):
        s.append(f'<circle class="iunit" cx="{700 + k * 30}" cy="66" r="8"/>')
    s.append('<text class="ann" x="530" y="124">E units: one per recorded E neuron, plus</text>'
             '<text class="ann" x="530" y="139">unrecorded E units to reach 4 E : 1 I</text>'
             '<text class="ann" x="688" y="96">I units: one per</text><text class="ann" x="688" y="109">recorded I neuron</text>'
             "<text class=\"ann\" x=\"530\" y=\"164\">Dale's law; I units project only within the network</text>"
             '<text class="ref" x="672" y="196" text-anchor="middle">recurrent network of rate units</text>')
    s.append('</svg>')

    cols = [('No context', 'none', False, False, False), ('Shared → all', 'shared', True, True, False),
            ('Shared → I', 'shared', False, True, False), ('Per-unit → I', 'unit', False, True, True),
            ('Per-unit → all', 'unit', True, True, True)]
    w2, h2 = 848, 136
    b = [f'<svg width="{w2}" height="{h2}" viewBox="0 0 {w2} {h2}" role="img" aria-label="Context routing variants">',
         '<text class="panel" x="4" y="14">b</text>']
    for k, (lab, fam, to_e, to_i, per_unit) in enumerate(cols):
        x0 = 40 + k * 162
        b.append(f'<text class="ann" x="{x0 + 50}" y="20" text-anchor="middle">{lab}</text>')
        b.append(f'<rect class="ctxsrc {fam}" x="{x0 + 34}" y="28" width="32" height="16" rx="3"/>')
        for j in range(3):
            b.append(f'<circle class="eunit" cx="{x0 + 14 + j * 22}" cy="96" r="7"/>')
        b.append(f'<circle class="iunit" cx="{x0 + 94}" cy="96" r="7"/>')
        targets = ([x0 + 14 + j * 22 for j in range(3)] if to_e else []) + ([x0 + 94] if to_i else [])
        for j, tx in enumerate(targets):
            width = (0.8 + (j * 1.7) % 3.2) if per_unit else 1.6
            b.append(f'<line class="ctxline {fam}" x1="{x0 + 50}" y1="44" x2="{tx}" y2="88" stroke-width="{width:.1f}"/>')
        note = {'shared': 'one weight per population', 'unit': 'own weight per unit', 'none': '—'}[fam]
        b.append(f'<text class="tick" x="{x0 + 50}" y="126" text-anchor="middle">{note}</text>')
    b.append('<text class="ref" x="844" y="112" text-anchor="end">○ E   ● I</text>')
    b.append('</svg>')
    return ''.join(s) + ''.join(b)


def strip_by_variant(values: dict, order, ceiling, title, ylab, yr=(-0.6, 1.0), w=848, h=240, ceiling_label=None,
                     mean_labels=True) -> str:
    """Dot strip: one column per variant, a dot per restart, a bar for the mean."""
    p = Plot(w, h, (-0.5, len(order) - 0.5), yr, ml=40, mb=48, mt=24)
    for t in np.arange(yr[0], yr[1] + 1e-9, 0.5):
        p.hline(round(t, 2), cls='zero' if abs(t) < 1e-9 else 'grid')
    if ceiling is not None and not np.isnan(ceiling):
        p.hline(ceiling, cls='ceil', label=ceiling_label or f'ceiling {ceiling:.2f}', anchor='start')
    for k, v in enumerate(order):
        label, fam = VARIANTS[v][:2]
        vals = values.get(v, [])
        for j, x in enumerate(vals):
            if not np.isnan(x):
                p.dot(k + (j - (len(vals) - 1) / 2) * 0.06, x, fam, f'{label}, restart {j + 1}: r = {x:+.2f}', r=4)
        m = mean(vals)
        if not np.isnan(m):
            p.add(f'<line class="meanbar {fam}" x1="{p.X(k - 0.3):.1f}" x2="{p.X(k + 0.3):.1f}" y1="{p.Y(m):.1f}" y2="{p.Y(m):.1f}">'
                  f'<title>{esc(label)}: mean r = {m:+.2f}</title></line>')
            if mean_labels:
                p.add(f'<text class="meanlabel" x="{p.X(k + 0.33):.1f}" y="{p.Y(m) + 4:.1f}">{m:+.2f}</text>')
        parts = label.split(' → ')
        p.add(f'<text class="tick" x="{p.X(k):.1f}" y="{h - 30}" text-anchor="middle">{esc(parts[0])}</text>')
        if len(parts) > 1:
            p.add(f'<text class="tick" x="{p.X(k):.1f}" y="{h - 17}" text-anchor="middle">→ {esc(parts[1])}</text>')
    p.yaxis([t for t in (-1.0, -0.5, 0, 0.5, 1.0) if yr[0] <= t <= yr[1]], '')
    p.add(f'<text class="ann" x="40" y="12">{esc(ylab)}</text>')
    return p.svg(title)


def fig_real(d):
    g = d['grid']
    order = [v for v in ('none_free', 'shared_free', 'sharedI_free', 'sharedE_free', 'freeI_free', 'freeE_free',
                         'free_free', 'freeI_free_qshift', 'free_free_qshift') if v in g]
    a = strip_by_variant({v: g[v]['r'] for v in order if v in g}, order, d['grid_ceiling'],
                         'Reward context per variant', 'held-out r with the recorded reward-context effect')
    sp = strip_by_variant({v: g[v]['spatial'] for v in order if v in g}, order, d['grid_spatial_ceiling'],
                          'Spatial fit per variant', 'r with the recorded spatial preference (fit check)',
                          yr=(0, 1.0), h=210, mean_labels=False)
    rows = [[VARIANTS[v][0], ' '.join(fmt(x) for x in g[v]['r']), fmt(mean(g[v]['r'])),
             ' '.join(fmt(x) for x in g[v]['spatial']), fmt(mean(g[v]['dll']), 1)] for v in order if v in g]
    tab = table(['Variant', 'Reward-context r per restart', 'Mean', 'Spatial r per restart', 'ΔLL ×1000 vs no context (mean)'], rows)

    panels = []
    for v in ('free_free', 'shared_free'):
        model, rec, is_e = d['scatter'][v]
        ok = ~np.isnan(model) & ~np.isnan(rec)
        r = np.corrcoef(model[ok], rec[ok])[0, 1]
        xl = float(np.ceil(np.nanmax(np.abs(model[ok]))))
        yl = float(np.ceil(np.nanmax(np.abs(rec[ok])) / 5) * 5)
        p = Plot(420, 260, (-xl, xl), (-yl, yl), ml=56, mt=24)
        yt = [t for t in (-10, -5, 0, 5, 10) if abs(t) <= yl]
        for t in yt:
            p.hline(t, cls='zero' if t == 0 else 'grid')
        p.vline(0, cls='zero')
        fam = VARIANTS[v][1]
        for xi, yi, e in zip(model[ok], rec[ok], is_e[ok]):
            p.dot(xi, yi, fam, f'{"E" if e else "I"} neuron: model {xi:+.2f}, recorded (test) {yi:+.2f} sp/s', r=3, hollow=not e)
        p.yaxis(yt, '')
        p.add(f'<text class="ann" x="56" y="12">{VARIANTS[v][0]}, best restart: r = {r:.2f} (n = {ok.sum()})</text>')
        p.xaxis([-xl, 0, xl], 'model effect (sp/s)')
        panels.append(p.svg(f'{v} scatter', ylabel_rot='recorded, test trials (sp/s)'))
    c = '<div class="grid2">' + ''.join(f'<div class="panelwrap"><span class="panel-l">{l}</span>{s}</div>'
                                         for l, s in zip('cd', panels)) + '</div>'
    return (f'<div class="panelwrap"><span class="panel-l">a</span>{a}</div>'
            f'<div class="panelwrap"><span class="panel-l">b</span>{sp}</div>{c}{tab}')


def fig_recovery_design() -> str:
    w, h = 848, 126
    s = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Recovery test design">',
         '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>']
    boxes = [(10, 'Teacher', 'a network already fit to 08_14,', 'so its mechanism is known'),
             (222, 'Synthetic data', 'Poisson spikes from the teacher', 'on the real trials and inputs'),
             (434, 'Students', '5 variants × 3 restarts,', 'fit exactly as on real data'),
             (646, 'Score', "held-out r with the teacher's", 'reward-context effects')]
    for x, t, l1, l2 in boxes:
        s.append(f'<rect class="inbox" x="{x}" y="14" width="190" height="74" rx="6"/>'
                 f'<text class="annb" x="{x + 95}" y="36" text-anchor="middle">{t}</text>'
                 f'<text class="ann" x="{x + 95}" y="56" text-anchor="middle">{l1}</text>'
                 f'<text class="ann" x="{x + 95}" y="71" text-anchor="middle">{l2}</text>')
    for x in (200, 412, 624):
        s.append(f'<line class="arrow" x1="{x + 2}" y1="51" x2="{x + 20}" y2="51" marker-end="url(#ah2)"/>')
    s.append('<text class="ref" x="105" y="108" text-anchor="middle">context weights × g (g = 1: recorded size; g = 3: larger)</text>')
    s.append('</svg>')
    return ''.join(s)


def fig_recovery(d) -> str:
    out, rows = [], []
    for ds in ('null', None, 'shared_g1', 'shared_g3', 'ext_g1', 'ext_g3', 'local_g1', 'local_g3'):
        if ds is None:
            out.append('<div></div>')
            continue
        label, fam, gain = TEACHERS[ds]
        if ds not in d['pilot']:
            out.append(f'<div class="pending"><p class="ann">{esc(label)}: <strong>still running</strong></p></div>')
            continue
        pd = d['pilot'][ds]
        svg = strip_by_variant(pd['students'], STUDENTS, pd['ceiling'], label,
                               f'{label}; teacher effect {pd["recorded_mean"]:+.2f} sp/s',
                               yr=(-1.0, 1.0), w=420, h=230, ceiling_label=f'ceiling {pd["ceiling"]:.2f}', mean_labels=False)
        out.append(f'<div class="panelwrap">{svg}</div>')
        rows.append([label, f'{pd["ceiling"]:.2f}'] + [' '.join(fmt(x) for x in pd['students'][v]) for v in STUDENTS])
    return ('<div class="grid2">' + ''.join(out) + '</div>'
            + table(['Teacher', 'Ceiling'] + [VARIANTS[v][0] for v in STUDENTS], rows))


def fig_calibrated(d) -> str:
    """Shared vs per-unit student r, real data vs teachers at g = 1."""
    items = [('Real data, 3 sessions', d['grid'].get('shared_free', {}).get('r', []), d['grid'].get('free_free', {}).get('r', []))]
    for ds in ('ext_g1', 'local_g1', 'null'):  # the shared teacher has no usable context effect (Section 6)
        if ds in d['pilot']:
            st = d['pilot'][ds]['students']
            items.append((TEACHERS[ds][0], st['shared_free'], st['free_free']))
    p = Plot(848, 54 + 38 * len(items), (-0.6, 1.0), (len(items) - 0.5, -0.5), ml=250, mt=30, mb=34)
    for t in (-0.5, 0, 0.5, 1.0):
        p.vline(t, cls='zero' if t == 0 else 'grid')
    rows = []
    for k, (lab, sh, pu) in enumerate(items):
        p.add(f'<text class="tick" x="{p.ml - 10}" y="{p.Y(k) + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
        for vals, fam, off in ((sh, 'shared', -0.13), (pu, 'unit', 0.13)):
            for x in vals:
                if not np.isnan(x):
                    p.dot(x, k + off, fam, f'{lab}, {FAMILY_LABEL[fam]} student: r = {x:+.2f}', r=3.5)
            m = mean(vals)
            if not np.isnan(m):
                p.add(f'<line class="meanbar {fam}" x1="{p.X(m):.1f}" x2="{p.X(m):.1f}" y1="{p.Y(k + off - 0.14):.1f}" '
                      f'y2="{p.Y(k + off + 0.14):.1f}"><title>{esc(lab)}: mean {m:+.2f}</title></line>')
        rows.append([lab, fmt(mean(sh)), fmt(mean(pu))])
    p.add('<text class="ann" x="250" y="14">students: <tspan class="t-shared">● shared → all</tspan>'
          '<tspan dx="14" class="t-unit">● per-unit → all</tspan><tspan dx="14">dots: restarts; bars: means</tspan></text>')
    p.xaxis([-0.5, 0, 0.5, 1.0], 'held-out r with the context effect')
    return p.svg('Calibrated comparison') + table(['Data', 'Shared student (mean r)', 'Per-unit student (mean r)'], rows)


# ---------------------------------------------------------------- page

CSS = """
:root { --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --muted:#8a8985; --grid:#e4e3df; --axis:#c9c8c3;
  --unit:#2a78d6; --shared:#eb6834; --none:#8a8985; --accent:#1baf7a; --panelbg:#f4f3f0; --ctx:#fbe3d8; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7; --muted:#8f8e88; --grid:#2e2e2c; --axis:#4a4945;
  --unit:#3987e5; --shared:#d95926; --none:#8f8e88; --accent:#199e70; --panelbg:#232321; --ctx:#3a2a22; } }
:root[data-theme="dark"] { --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7; --muted:#8f8e88; --grid:#2e2e2c; --axis:#4a4945;
  --unit:#3987e5; --shared:#d95926; --none:#8f8e88; --accent:#199e70; --panelbg:#232321; --ctx:#3a2a22; }
body { background:var(--surface); color:var(--text); font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; margin:0; }
main { max-width:880px; margin:0 auto; padding:28px 16px 80px; }
h1 { font-size:1.6em; line-height:1.25; margin:0 0 4px; } h2 { font-size:1.2em; margin:2.2em 0 .4em; }
h3 { font-size:1em; margin:1.6em 0 .3em; }
.meta { color:var(--text2); font-size:.9em; margin-bottom:1.4em; }
.abstract { background:var(--panelbg); border-radius:8px; padding:14px 18px; }
.abstract p { margin:.5em 0; }
figure { margin:1.4em 0 2em; } figcaption { font-size:.92em; color:var(--text2); margin-top:.5em; }
figcaption strong { color:var(--text); }
svg { max-width:100%; height:auto; display:block; overflow:visible; }
svg text { fill:var(--text2); font-size:11px; font-family:inherit; }
svg .panel { font-weight:700; fill:var(--text); font-size:13px; }
svg .annb { font-weight:600; fill:var(--text); }
svg .axlabel { fill:var(--text2); } svg .tick { fill:var(--muted); font-variant-numeric:tabular-nums; }
svg .ref { fill:var(--muted); font-size:10px; }
.grid { stroke:var(--grid); stroke-width:1; } .zero { stroke:var(--axis); stroke-width:1.2; }
.axis { stroke:var(--axis); stroke-width:1; } .diag { stroke:var(--axis); stroke-width:1; }
.ceil { stroke:var(--accent); stroke-width:1.5; } .refline { stroke:var(--muted); stroke-width:1; }
.mk { stroke:var(--surface); stroke-width:1.5; } .mk.hollow { fill:none !important; stroke-width:1.5; }
.mk.unit { fill:var(--unit); } .mk.shared { fill:var(--shared); } .mk.none { fill:var(--none); }
.mk.hollow.unit { stroke:var(--unit); } .mk.hollow.shared { stroke:var(--shared); } .mk.hollow.none { stroke:var(--none); }
.mk:hover { stroke:var(--text); }
.meanbar { stroke-width:3; stroke-linecap:round; } .meanbar.unit { stroke:var(--unit); } .meanbar.shared { stroke:var(--shared); } .meanbar.none { stroke:var(--none); }
.meanline { stroke:var(--text); stroke-width:1.5; } svg .meanlabel { fill:var(--text2); font-size:10.5px; }
.t-unit { fill:var(--unit); } .t-shared { fill:var(--shared); }
.loc { fill:none; stroke:var(--text2); stroke-width:1.5; } .locnum { fill:var(--text); font-weight:600; font-size:12px; }
.fix { fill:var(--text); }
.segA { fill:var(--unit); opacity:.85; } .segB { fill:var(--none); opacity:.55; }
.win.pre { fill:var(--ctx); } .win.vis { fill:var(--panelbg); stroke:var(--axis); }
.winlabel { fill:var(--text); font-size:10.5px; }
.evt { stroke:var(--text2); stroke-width:1.2; }
.bar.excluded { fill:var(--grid); } .bar.kept { fill:var(--text2); }
.inbox { fill:var(--panelbg); stroke:var(--axis); } .ctxbox { fill:var(--ctx); stroke:var(--axis); }
.net { fill:none; stroke:var(--axis); stroke-width:1.2; }
.eunit { fill:var(--panelbg); stroke:var(--text2); stroke-width:1.2; } .iunit { fill:var(--text2); }
.arrow { stroke:var(--muted); stroke-width:1.2; } .arrowhead { fill:var(--muted); }
.ctxsrc.shared { fill:var(--shared); } .ctxsrc.unit { fill:var(--unit); } .ctxsrc.none { fill:var(--grid); }
.ctxline.shared { stroke:var(--shared); } .ctxline.unit { stroke:var(--unit); }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px 18px; align-items:start; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px 14px; align-items:start; }
@media (max-width:640px) { .grid2, .grid3 { grid-template-columns:1fr; } }
.panelwrap { position:relative; } .panel-l { position:absolute; left:2px; top:0; font-weight:700; font-size:13px; }
.pending { background:var(--panelbg); border-radius:6px; padding:8px 12px; }
details { margin:.4em 0 0; font-size:.88em; color:var(--text2); } summary { cursor:pointer; }
table { border-collapse:collapse; margin:.5em 0; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--grid); padding:3px 10px 3px 0; text-align:left; }
.tag { display:inline-block; font-size:.75em; font-weight:700; border-radius:4px; padding:0 5px; margin-right:6px;
  vertical-align:1px; background:var(--panelbg); color:var(--text2); border:1px solid var(--axis); }
.gap { border-left:3px solid var(--shared); padding:4px 12px; background:var(--panelbg); border-radius:0 6px 6px 0; margin:.8em 0; }
.legendrow { font-size:.88em; color:var(--text2); margin:.3em 0; }
.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 12px; vertical-align:0; }
code { font-size:.9em; }
"""


def main():
    d = gather()
    fig2, split_r = fig_measure(d)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    g, pl = d['grid'], d['pilot']
    gm = lambda v, k='r': mean(g.get(v, {}).get(k, []))
    pm = lambda ds, v: mean(pl[ds]['students'][v]) if ds in pl else np.nan
    n_e = sum(c['E'] for c in d['counts'].values())
    n_i = sum(c['I'] for c in d['counts'].values())
    n_tr = sum(c['verified'] for c in d['counts'].values())
    n_ex = sum(c['exported'] for c in d['counts'].values())
    n_sw = sum(c['reward_switches'] for c in d['counts'].values())
    have_shared = 'shared_g1' in pl
    have_shift = 'free_free_shift50' in g
    legend = ('<p class="legendrow">Colour code in every figure:<span class="sw" style="background:var(--none)"></span>no context'
              '<span class="sw" style="background:var(--shared)"></span>shared context'
              '<span class="sw" style="background:var(--unit)"></span>per-unit context.'
              ' Filled markers: E neurons or units; open: I.</p>')

    body = f"""
<h1>Recovery pilot: can a model with one shared reward signal reproduce SC reward-context modulation?</h1>
<div class="meta">Report on <code>results/recovery_pilot</code> and the verified-trial fits it calibrates
(<code>results/verified_grid</code>). Newton SC sessions 08_13–08_15. Built {esc(__import__('datetime').date.today())} from
commit <code>{esc(commit)}</code> by <code>scripts/make_recovery_report.py</code>. Prepared by Claude for James Herman.</div>

<section class="abstract">
<p>@@FOREWORD@@</p>
<p>@@SUMMARY@@</p>
</section>

{legend}

<h2>1. The data: what the monkey did and which trials were analysed</h2>
<p><span class="tag">S</span>The monkey fixated, then made a saccade to one target at one of four locations. Two
context variables changed across a session in blocks (Fig. 1a, b). The <em>reward map</em> set which pair of
locations gave the large reward (1 + 2, or 3 + 4). The <em>probability block</em> set which location was the frequent
target (1 or 3). The reward map switched every 48 successful trials,
{min(c['reward_switches'] for c in d['counts'].values())}–{max(c['reward_switches'] for c in d['counts'].values())} times per session
({n_sw} switches over three sessions); the probability block switched half as often.</p>
<p><span class="tag">S</span>We analysed {n_tr} of {n_ex} exported trials: the successful ones whose event times were verified
against the task-control records (Fig. 1d). The original export also contained fixation breaks and trials with
unreliable event times, and its reward times were 0.46–0.65 s late (trial audit by the pipeline session,
<code>data/trial_audit</code>). The {n_e + n_i} neurons are {n_e} classic SC neurons, fit by excitatory (E) model units,
and {n_i} putative interneurons, fit by inhibitory (I) units. That classification comes from the analysis pipeline.</p>
<figure>{fig_task(d)}
<figcaption><strong>Figure 1. Reward context changes in blocks, and the analysed data are the verified successful trials.</strong>
(a) Task layout and the two context variables. (b) Context over session 08_15: blue segments are reward map A
(locations 1 + 2 large) and probability block 1 (location 1 frequent); gray segments are the alternatives.
(c) Trial epochs; the context effect analysed here is measured in the pre-target window, before any target is
visible, so it can reflect only the block context. (d) Trials kept per session (solid) out of those exported (faint).</figcaption></figure>

<h2>2. The quantity: a neuron's reward-context effect, and how reliable it is</h2>
<p><span class="tag">S</span>For each neuron and each reward-map switch, we took its mean firing rate in the
pre-target window (−300 to 0 ms) over up to 30 trials after the switch, minus the same before it (Fig. 2a). The
sign is positive when the neuron's preferred location became the large-reward location. Averaging over switches
gives one number per neuron, its <em>reward-context effect</em>. Preferred location is the location of the neuron's
largest visual response.</p>
<p><span class="tag">F</span>The effect differs between neurons, and that pattern is reliable. Neuron-to-neuron
differences on 15% held-out test trials correlate with those on the remaining trials at r = {split_r:.2f}
(Fig. 2c), which is the ceiling any model can reach on this test split. Across neurons the mean effect is small,
{np.nanmean(d['effects']):+.2f} sp/s (Fig. 2b). This report is about the <em>pattern</em> across neurons, not the mean.</p>
<figure>{fig2}
<figcaption><strong>Figure 2. Reward context shifts pre-target firing by different amounts in different neurons, reliably.</strong>
(a) Example E neuron (08_15, prefers location 1) around one switch: pre-target rate per trial, blue when its preferred
location gives the large reward; lines are the before and after means. Its effect averaged over the session's 11
switches is {d['example']['neuron_effect']:+.1f} sp/s. (b) Effects of all {n_e + n_i} neurons (filled E, open I).
(c) Reliability: effect on the held-out test trials (y) against the other trials (x), one dot per neuron;
the correlation is the ceiling for the model comparisons below.</figcaption></figure>

<h2>3. The models: what differs between variants, and that they fit</h2>
<p><span class="tag">S</span>Each model is a recurrent network of rate units obeying Dale's law (Fig. 3a). Every recorded
neuron is fit by its own unit, E neurons by E units and I neurons by I units, and the network is padded with
unrecorded E units to a 4 : 1 E : I ratio. The network receives the task events and eye position, three slow-drift
inputs (smooth functions of trial number that absorb slow changes in firing over the session), and in most variants
two context inputs. Variants differ only in how the context inputs reach the units (Fig. 3b): not at all; through
one shared weight per population (all E units get the same drive, as do all I units); or through a separate weight
for each unit. A shared input can still affect neurons differently, because the network's own connections route it.</p>
<p><span class="tag">S</span>Fitting adjusts all weights so the model's firing matches the recordings: trial-averaged
responses per condition, single-trial variability, and mean pre-target and visual-window rates per location and
context state. Trials were split once into fitting (70%), checkpoint selection (15%) and test (15%) sets, the same
for every variant; each variant was fit 6 times from different random starts ("restarts"). Every r reported here
is on the test trials.</p>
<figure>{fig_models()}
<figcaption><strong>Figure 3. The variants differ only in how the two context inputs reach the network.</strong>
(a) Network and inputs. (b) Context routing; line widths stand for weights: equal for shared routing, free for
per-unit routing.</figcaption></figure>

<h2>4. Real data: only per-unit context reproduces which neurons are modulated</h2>
<p><span class="tag">F</span>All variants fit the spatial tuning of the neurons well (Fig. 4b: mean r = {min(gm(v, 'spatial') for v in VARIANTS if v in g):.2f}–{max(gm(v, 'spatial') for v in VARIANTS if v in g):.2f}),
so the comparison is between models that fit the dominant structure in the data. They differ in the reward-context
pattern (Fig. 4a). With per-unit context, models reached r = {gm('free_free'):.2f} (to all units), {gm('freeI_free'):.2f}
(I units only) and {gm('freeE_free'):.2f} (E units only), against a ceiling of {d['grid_ceiling']:.2f}. With shared
context they reached {min(gm(v) for v in ('shared_free', 'sharedI_free', 'sharedE_free')):+.2f} to {max(gm(v) for v in ('shared_free', 'sharedI_free', 'sharedE_free')):+.2f}, no better than without context
({gm('none_free'):+.2f}). Fig. 4c–d show the neuron-by-neuron agreement for the best restart of each.</p>
<p><span class="tag">S</span>Held-out likelihood does not separate the variants: differences are a few thousandths of a
log-unit per neuron per time bin, the same size as differences between restarts (table). The extra per-unit weights
therefore do not improve the overall fit; they matter only for this small effect.</p>
<figure>{fig_real(d)}
<figcaption><strong>Figure 4. On real data, per-unit context reproduces the neuron-by-neuron pattern of reward
modulation and shared context does not.</strong> (a) Held-out r between each model's reward-context effects and the
recorded ones on test trials, pooled over the {n_e + n_i} neurons of three sessions; one dot per restart, bar = mean,
green line = reliability ceiling; the gray columns at right are the control with the context inputs shifted
out of register with the trials (Section 7). (b) The same for spatial preference, as a check that all variants fit.
(c, d) Per-neuron effects for the best restart (highest selection-set likelihood) of per-unit and shared variants.</figcaption></figure>

<h2>5. Why a recovery test, and how it works</h2>
<p><span class="tag">S</span>Figure 4 alone cannot show that a single shared reward signal is insufficient. A shared-context
model might be able to produce the pattern and simply not have found it during fitting. Conversely, per-unit models
have more free weights, and might match any reliable per-neuron pattern, including one unrelated to reward. A
recovery test asks what the fitting procedure returns when the truth is known (Fig. 5). We took networks already
fit to session 08_14 as <em>teachers</em>, generated spikes from them on the same trials, and fit <em>students</em>
to those spikes exactly as on real data. Teachers differ in mechanism: no context; shared context; per-unit context
to all units; per-unit context to I units only, so that any neuron-specific effect on E units must pass through the
I→E connections. Multiplying a teacher's context weights by g = 3 makes its effects larger; g = 1 keeps them close to
the recorded size.</p>
<figure>{fig_recovery_design()}
<figcaption><strong>Figure 5. The recovery test fits students to synthetic data from teachers whose mechanism is
known.</strong> Scores use the same held-out r as Fig. 4, but against the teacher-generated effects; the ceiling is
again the test-versus-rest reliability, now of the synthetic data.</figcaption></figure>

<h2>6. Recovery results</h2>
@@RECOVERY_TEXT@@
<figure>{fig_recovery(d)}
<figcaption><strong>Figure 6. What students recover from each teacher</strong> (session 08_14, 41 neurons, 3 restarts
per student). Each panel is one teacher; dots are restarts, bars means, green lines the ceilings. With 41 neurons,
r below about ±0.3 is within single-session noise (see the no-context teacher).</figcaption></figure>

<h2>7. The calibrated comparison</h2>
@@CALIBRATED_TEXT@@
<figure>{fig_calibrated(d)}
<figcaption><strong>Figure 7. Shared and per-unit students on real data and on teachers of realistic effect size.</strong>
Real-data values pool three sessions (109 neurons, 6 restarts); teacher values are one session (41 neurons, 3 restarts).</figcaption></figure>

<h2>8. What this does not show yet, and what comes next</h2>
@@GAPS_TEXT@@
"""
    body = body.replace('@@FOREWORD@@', FOREWORD).replace('@@SUMMARY@@', summary_text(d, gm, pm))
    body = body.replace('@@RECOVERY_TEXT@@', recovery_text(d, pm)).replace('@@CALIBRATED_TEXT@@', calibrated_text(d, gm, pm))
    body = body.replace('@@GAPS_TEXT@@', gaps_text(d))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Recovery Pilot Report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>')
    OUT.write_text(page, encoding='utf-8')
    print(f"Wrote {OUT} ({len(page) / 1024:.0f} KB)")


FOREWORD = (
    "SC neurons fire differently before the target appears depending on which locations currently give a large "
    "reward, and the size of this reward-context effect differs reliably between neurons. If reward context reaches "
    "SC as one global signal, the local excitatory–inhibitory circuit must turn it into these neuron-specific "
    "effects; if it arrives already specific to each neuron, the circuit need not. When we fit E–I network models to "
    "the recordings, only models with a separate context input for each unit reproduced the neuron-by-neuron pattern. "
    "Because that contrast could reflect fitting limits rather than mechanism, we fit the same models to synthetic "
    "data from networks whose mechanism is known. This report presents that recovery pilot, one session, and what it "
    "does and does not establish.")


def summary_text(d, gm, pm) -> str:
    s = (f"On real data, per-unit context models reproduced the pattern (held-out r = {gm('free_free'):.2f}, "
         f"ceiling {d['grid_ceiling']:.2f}); shared-context models did not ({gm('shared_free'):+.2f}). In the pilot, "
         f"students recovered per-unit teachers' patterns, so the test can succeed. Shared-context students recovered much "
         f"of a per-unit teacher's pattern when effects were large, but erratically at realistic effect sizes "
         f"({pm('ext_g1', 'shared_free'):+.2f} and {pm('local_g1', 'shared_free'):+.2f} for two such teachers). ")
    if 'shared_g1' in d['pilot']:
        s += ("A shared-context teacher, the control for whether the test can confirm a shared mechanism, turned out "
              "to have almost no reward-context effect of its own, so that control remains open. ")
    g = d['grid']
    if 'free_free_qshift' in g:
        s += (f"Per-unit students scored above zero even on a teacher with no context, raising the worry that their "
              f"real-data success was generic. A control on the real data rules most of this out: with the context "
              f"inputs shifted out of register with the trials, per-unit models fell from r = {gm('free_free'):.2f} to "
              f"{gm('free_free_qshift'):.2f} (all units) and from {gm('freeI_free'):.2f} to {gm('freeI_free_qshift'):+.2f} "
              f"(I units only). The neuron-specific reward-context pattern is therefore genuinely carried by context "
              f"that is aligned with the task, and per-unit context input reproduces it. Whether a single shared signal "
              f"could also reproduce it remains untested, because the pilot has not yet shown that the procedure "
              f"recovers a shared mechanism when that is the truth. ")
    else:
        s += ("Per-unit students also scored above zero on a teacher with no context at all, so part of their real-data "
              "advantage may be spurious. The real-data contrast is therefore suggestive, not yet decisive. ")
    s += "The full test, on all re-exported Newton and Feynman sessions, is specified in Section 8."
    return s


def recovery_text(d, pm) -> str:
    pl = d['pilot']
    t = []
    t.append(f"<p><span class=\"tag\">F</span>The test can succeed: per-unit students recovered per-unit teachers at "
             f"realistic size, r = {pm('ext_g1', 'free_free'):.2f} (all-unit teacher) and {pm('local_g1', 'freeI_free'):.2f} "
             f"(I-only teacher, I-only student), at or near the ceilings ({pl['ext_g1']['ceiling']:.2f}, {pl['local_g1']['ceiling']:.2f}). "
             f"Students without context scored at or below zero on every context teacher, as they should.</p>")
    t.append(f"<p><span class=\"tag\">F</span>A shared signal can make neuron-specific effects. With large effects (g = 3), shared "
             f"students recovered {pm('ext_g3', 'shared_free'):.2f} of an all-unit teacher's pattern and "
             f"{pm('local_g3', 'shared_free'):.2f} of an I-only teacher's: the network routes one common input into "
             f"different effects in different neurons. At realistic size (g = 1) the same students were erratic: "
             f"{' '.join(fmt(x) for x in pl['ext_g1']['students']['shared_free'])} and "
             f"{' '.join(fmt(x) for x in pl['local_g1']['students']['shared_free'])} across restarts.</p>")
    if 'shared_g1' in pl:
        t.append(f"<p><span class=\"tag\">F</span>The shared-context teacher could not serve as the shared-mechanism control. "
                 f"It came from the shared-context fit to 08_14, which had learned almost no context effect: at g = 1 its "
                 f"mean effect was {pl['shared_g1']['recorded_mean']:+.2f} sp/s and its per-neuron pattern was barely "
                 f"reliable (ceiling {pl['shared_g1']['ceiling']:.2f}). At g = 3 the pattern became reliable (ceiling "
                 f"{pl['shared_g3']['ceiling']:.2f}), but students without any context input reproduced it as well "
                 f"({pm('shared_g3', 'none_free'):+.2f}) as shared students ({pm('shared_g3', 'shared_free'):+.2f}), so it "
                 f"reflects the teacher's drift and task structure rather than context.</p>")
    t.append(f"<p><span class=\"tag\">F</span>Per-unit students can score above zero without any reward context. On the "
             f"no-context teacher, the effect measure was itself reliable (ceiling {pl['null']['ceiling']:.2f}), "
             f"because the teacher's slow-drift inputs change firing over the session, and switches happen at particular "
             f"times. Per-unit students matched this drift-driven pattern with r = "
             f"{' '.join(fmt(x) for x in pl['null']['students']['free_free'])} (all units) and "
             f"{' '.join(fmt(x) for x in pl['null']['students']['freeI_free'])} (I only), using their context inputs as "
             f"extra slow regressors.</p>")
    return ''.join(t)


def calibrated_text(d, gm, pm) -> str:
    g = d['grid']
    t = (f"<p><span class=\"tag\">F</span>The real-data result is weaker evidence than it first appeared. On real data the "
         f"shared student scored {gm('shared_free'):+.2f} against {gm('free_free'):+.2f} for the per-unit student. On a "
         f"per-unit teacher of realistic size the gap was smaller ({pm('ext_g1', 'shared_free'):+.2f} vs "
         f"{pm('ext_g1', 'free_free'):+.2f}) for one teacher and similar ({pm('local_g1', 'shared_free'):+.2f} vs "
         f"{pm('local_g1', 'free_free'):+.2f}) for the other (Fig. 7). So a large gap is what this procedure produces "
         f"for at least some data generated by neuron-specific input, consistent with the real-data result, but it is "
         f"not unique to such data: on the no-context teacher the per-unit student also outscored the shared one "
         f"({pm('null', 'free_free'):+.2f} vs {pm('null', 'shared_free'):+.2f}).</p>")
    if 'free_free_qshift' in g:
        ctl = (f"<p><span class=\"tag\">F</span>On real data, the per-unit models' success depends on correctly aligned "
               f"context, so it is not the generic advantage seen on the no-context teacher. In a control, each context "
               f"input was shifted by a quarter of its own period (24 trials for the reward map, 48 for the probability "
               f"block): it keeps the same switching statistics but is uncorrelated with the true context. Per-unit "
               f"models given these inputs reached r = {gm('free_free_qshift'):+.2f} (all units; restarts "
               f"{min(g['free_free_qshift']['r']):+.2f} to {max(g['free_free_qshift']['r']):+.2f}) and "
               f"{gm('freeI_free_qshift'):+.2f} (I only; {min(g['freeI_free_qshift']['r']):+.2f} to "
               f"{max(g['freeI_free_qshift']['r']):+.2f}), against {gm('free_free'):+.2f} and {gm('freeI_free'):+.2f} "
               f"with aligned context (Fig. 4a, gray columns); the ranges do not overlap. About "
               f"{gm('free_free_qshift'):.2f} of the all-unit score is what extra slow inputs buy without real context.</p>")
        t = ctl + t.replace("The real-data result is weaker evidence than it first appeared. ",
                            "What the pilot cannot yet say is whether a shared signal could have produced the "
                            "real-data pattern. ")
    return t


def gaps_text(d) -> str:
    g, pl = d['grid'], d['pilot']
    items = []
    if 'shared_g1' not in pl:
        items.append("<strong>Shared-teacher control.</strong> Whether shared students recover a teacher that really "
                     "uses a shared signal is running now (<code>results/recovery_pilot/shared_g*</code>). Without it, "
                     "the test has not shown that it can confirm the shared mechanism when that mechanism is true.")
    else:
        items.append("<strong>Shared-teacher control still open.</strong> The teacher taken from a shared-context fit had "
                     "almost no context effect (Section 6), so the test has not yet shown that it can confirm a shared "
                     "mechanism when that is the truth. A usable shared teacher needs a sizeable, genuinely "
                     "context-driven effect: for example, a shared-context network trained with the context effect "
                     "weighted more heavily, checked against a no-context student before use.")
    if 'free_free_qshift' not in g:
        items.append("<strong>Shifted-context control on real data.</strong> Per-unit models fit with each context input "
                     "shifted by a quarter of its period, which leaves it uncorrelated with the true context, are running "
                     "now (<code>results/verified_grid/*_qshift</code>). If they reach r near 0.43 too, the real-data "
                     "result reflects slow structure aligned with the switches, not reward context.")
    items += [
        "<strong>One session, three restarts.</strong> The pilot used 41 neurons from 08_14. Single-session r varies by "
        "about ±0.4 between restarts (Fig. 6), so the pilot calibrates the test but cannot settle it.",
        "<strong>Teachers are themselves fits.</strong> Their mechanisms are realistic only to the extent that the fits are; "
        "a teacher whose context effects match the recorded pattern exactly does not exist yet.",
        "<strong>Drift and context are partly confounded</strong> in this design, because switches happen at fixed "
        "intervals. The effect measure picks up slow drift (no-context teacher ceiling "
        f"{pl['null']['ceiling']:.2f}).",
        "<strong>E versus I as the landing site</strong> of neuron-specific input is only weakly identifiable at "
        "realistic effect sizes: each teacher was recovered best by the matching student, but by small margins.",
    ]
    out = ''.join(f'<div class="gap"><span class="tag">Gap</span>{x}</div>' for x in items)
    out += ("<p><span class=\"tag\">S</span>Next, once the corrected exports arrive (Newton and Feynman, all sessions, "
            "outcome-filtered trials): refit all variants with 6 restarts; run the recovery test on every session with "
            "shared, all-unit, I-only and no-context teachers at realistic effect size; run the shifted-context control "
            "on every session; report the shared/per-unit gap on real data against its distribution across teachers.</p>")
    return out


if __name__ == '__main__':
    main()
