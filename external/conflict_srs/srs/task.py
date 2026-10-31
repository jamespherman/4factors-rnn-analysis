"""
srsClassic task: trial tables and network inputs.

TRIAL TABLE (the single interface; dict of 1-D numpy arrays or a pandas DataFrame, one row per trial):
  trial_type   0 = single-target (instruction), 1 = two-target choice
  id_L, id_R   identity at the left / right location: 1 = T1, 2 = T2, 0 = absent
  sal_L, sal_R salience level (0 low, 1 high; floats allowed); ignored where absent
  rew_L, rew_R reward (ms) delivered if that target is chosen (per-trial value incl. jitter); 0 where absent
  V1, V2       value belief (ms) for T1 / T2 on this trial (supplied externally or by belief_delta_rule)
  side_bias    per-trial side-bias term (model units; > 0 favours the RIGHT location). Default 0.
  fix_dur      fixation duration before target onset (ms)
  delay        target onset -> go (ms); recorded task 450. Negative = go before targets (time-pressure redesign).
  onset_L, onset_R  extra onset offset of each target (ms; onset-asynchrony redesign). Default 0.
Any extra columns (block, trial, session, ...) are passed through by simulate().

Recorded-task facts (bilateral-SC-recordings, PLDAPS srsClassic_Moretraining dictionary): fixation 500-800 ms,
targets on with fixation and stay on, fixed 450-ms delay, go = fixation off, 450-ms response window; two fixed
locations (one per hemifield); identity side random per trial; reward tied to identity, rich identity reverses at
every unsignalled block; per-block means U[37,191] ms x k with separation >= 40k, trial jitter N(0, (14k)^2), cap 400;
each block = 50 single-target rows (25 T1 + 25 T2, random order, failed rows repeated) then 12-48 choice rows
(balanced congruent/conflict x side); salience = isoluminant hue contrast, 2 levels, random on single-target rows.
"""

import numpy as np
import torch

COLS = ["trial_type", "id_L", "id_R", "sal_L", "sal_R", "rew_L", "rew_R", "V1", "V2", "side_bias",
        "fix_dur", "delay", "onset_L", "onset_R", "engaged"]
DEFAULTS = dict(side_bias=0.0, onset_L=0.0, onset_R=0.0, delay=450.0, engaged=1.0)
# engaged (mixture/lapse route): 1 = value belief delivered as is; 0 = both identities get the mean belief
# (value carries no difference on that trial, so choice falls to side bias / salience / noise); fractions allowed.


def from_recorded(path_or_df, dt=10.0, keep_extra=True, completed_only=True):
    """Convert a recorded srsClassic trial table (bilateral-SC-recordings export with belief_t1_ms / belief_t2_ms,
    e.g. 4factors-RNN results/hub_models/belief/*_trials_with_belief.csv) into a model trial table.
    Side codes in the export: 1 = RIGHT (x > 0), 2 = LEFT (checked against saccade endpoints on 08-19 and 08-20).
    t1_side / t2_side give each identity's side on that trial; on single-target trials only shown_target_id is shown.
    Salience: contrast 180 -> 1 (high), 20 -> 0 (low). Timing from t_fix_acquired / t_target_on / t_go_signal (s).
    The animal's choice is kept as monkey_choice_loc (0 L / 1 R / -1 none)."""
    import pandas as pd
    d = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    # Failed/aborted rows have no outcome, chosen_side 0, and (single-target) no shown identity or timing.
    completed = d["outcome"].notna() & d["chosen_side"].isin([1, 2])
    if completed_only:
        d = d[completed].reset_index(drop=True)
        completed = completed[completed].reset_index(drop=True)
    n = len(d)
    fix_med = float(np.nanmedian((d["t_target_on"] - d["t_fix_acquired"]).to_numpy() * 1000))
    del_med = float(np.nanmedian((d["t_go_signal"] - d["t_target_on"]).to_numpy() * 1000))
    choice = d["is_choice_trial"].to_numpy().astype(float)
    shown = d["shown_target_id"].fillna(0).to_numpy()
    t1L = d["t1_side"].to_numpy() == 2
    t2L = d["t2_side"].to_numpy() == 2
    present1 = (choice == 1) | (shown == 1)
    present2 = (choice == 1) | (shown == 2)
    id_L = np.where(present1 & t1L, 1, np.where(present2 & t2L, 2, 0)).astype(float)
    id_R = np.where(present1 & ~t1L, 1, np.where(present2 & ~t2L, 2, 0)).astype(float)
    hs1 = (d["t1_salience_contrast"].to_numpy() >= 90).astype(float)
    hs2 = (d["t2_salience_contrast"].to_numpy() >= 90).astype(float)
    r1, r2 = d["t1_reward_ms"].to_numpy().astype(float), d["t2_reward_ms"].to_numpy().astype(float)
    pick = lambda idv, a, b: np.where(idv == 1, a, np.where(idv == 2, b, 0.0))
    fix = np.nan_to_num((d["t_target_on"] - d["t_fix_acquired"]).to_numpy() * 1000, nan=fix_med)
    delay = np.nan_to_num((d["t_go_signal"] - d["t_target_on"]).to_numpy() * 1000, nan=del_med)
    cs = d["chosen_side"].to_numpy()
    t = dict(trial_type=choice, id_L=id_L, id_R=id_R, sal_L=pick(id_L, hs1, hs2), sal_R=pick(id_R, hs1, hs2),
             rew_L=pick(id_L, r1, r2), rew_R=pick(id_R, r1, r2),
             V1=d["belief_t1_ms"].to_numpy().astype(float), V2=d["belief_t2_ms"].to_numpy().astype(float),
             side_bias=np.zeros(n), fix_dur=np.round(fix / dt) * dt, delay=np.round(delay / dt) * dt,
             onset_L=np.zeros(n), onset_R=np.zeros(n), engaged=np.ones(n),
             monkey_choice_loc=np.where(cs == 1, 1, np.where(cs == 2, 0, -1)).astype(float),
             monkey_rt_ms=d["rt_s"].to_numpy() * 1000, monkey_completed=completed.to_numpy().astype(float))
    if keep_extra:
        for c in ("trial_row", "block_type", "is_conflict", "high_reward_side", "high_salience_side",
                  "block_mean_t1_ms", "block_mean_t2_ms"):
            if c in d:
                t[c] = d[c].to_numpy()
    return t


# ----------------------------------------------------------------------------- table utilities
def as_table(table):
    """Accept dict / DataFrame / csv path; return dict of numpy arrays with all required columns."""
    if isinstance(table, str):
        import pandas as pd
        table = pd.read_csv(table)
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        table = {c: table[c].to_numpy() for c in table.columns}
    t = {k: np.asarray(v) for k, v in table.items()}
    n = len(next(iter(t.values())))
    for c in COLS:
        if c not in t:
            if c in DEFAULTS:
                t[c] = np.full(n, DEFAULTS[c], float)
            else:
                raise KeyError(f"trial table is missing column '{c}'")
    for c in COLS:
        t[c] = t[c].astype(float)
    return t


def subset(table, idx):
    return {k: v[idx] for k, v in table.items()}


# ----------------------------------------------------------------------------- block values
def draw_block_means(rng, k=1.0, lo=37.0, hi=191.0, min_sep=40.0):
    while True:
        a, b = rng.uniform(lo, hi, 2) * k
        if abs(a - b) >= min_sep * k:
            return max(a, b), min(a, b)


def trial_reward(rng, mean, k, n=None):
    return np.clip(mean + rng.normal(0, 14.0 * k, n), 0, 400)


# ----------------------------------------------------------------------------- synthetic training trials
def make_training_trials(cfg, n, rng, stage="S3"):
    """Independent trials for training. Stage S1: single-target only, flat reward (learn fixation + saccade).
    S2: real block values, beliefs = block means. S3: beliefs = block means + N(0, belief_noise)."""
    k = rng.choice([1.0, 1.5], n)
    rich = np.zeros(n)
    poor = np.zeros(n)
    for i in range(n):
        rich[i], poor[i] = draw_block_means(rng, k[i])
    t1_rich = rng.random(n) < 0.5
    m1 = np.where(t1_rich, rich, poor)
    m2 = np.where(t1_rich, poor, rich)
    choice = (rng.random(n) < (0.0 if stage == "S1" else cfg.p_choice)).astype(float)
    t1_left = rng.random(n) < 0.5
    id_L = np.where(t1_left, 1, 2).astype(float)
    id_R = np.where(t1_left, 2, 1).astype(float)
    sal_hi_left = rng.random(n) < 0.5
    sal_L = sal_hi_left.astype(float)
    sal_R = 1.0 - sal_L
    # single-target: show one identity on a random side, salience random and independent
    single = choice == 0
    shown = rng.integers(1, 3, n).astype(float)
    side_R = rng.random(n) < 0.5
    id_L = np.where(single, np.where(side_R, 0, shown), id_L)
    id_R = np.where(single, np.where(side_R, shown, 0), id_R)
    sal_rand = (rng.random(n) < 0.5).astype(float)
    sal_L = np.where(single, np.where(side_R, 0, sal_rand), sal_L)
    sal_R = np.where(single, np.where(side_R, sal_rand, 0), sal_R)
    r1 = trial_reward(rng, m1, k, n)
    r2 = trial_reward(rng, m2, k, n)
    if stage == "S1":
        r1 = r2 = np.full(n, cfg.value_scale)
    rew_L = np.where(id_L == 1, r1, np.where(id_L == 2, r2, 0.0))
    rew_R = np.where(id_R == 1, r1, np.where(id_R == 2, r2, 0.0))
    if stage == "S1":
        V1, V2 = np.full(n, 150.0), np.full(n, 150.0)
    elif stage == "S2":
        V1, V2 = m1.copy(), m2.copy()
    else:
        V1 = np.clip(m1 + rng.normal(0, cfg.belief_noise, n), 0, 400)
        V2 = np.clip(m2 + rng.normal(0, cfg.belief_noise, n), 0, 400)
    fix = rng.uniform(cfg.fix_min, cfg.fix_max, n)
    fix = np.round(fix / cfg.dt) * cfg.dt
    return dict(trial_type=choice, id_L=id_L, id_R=id_R, sal_L=sal_L, sal_R=sal_R, rew_L=rew_L, rew_R=rew_R,
                V1=V1, V2=V2, side_bias=np.zeros(n), fix_dur=fix, delay=np.full(n, cfg.delay),
                onset_L=np.zeros(n), onset_R=np.zeros(n), mean1=m1, mean2=m2, engaged=np.ones(n))


# ----------------------------------------------------------------------------- recorded-like sessions
def make_session(cfg, rng, n_blocks=8, k=1.5, n_choice=(12, 48), alpha=0.1, belief_noise=0.0, v0=None):
    """A session with the recorded block structure (no failures/repeats; for trial-matched simulation use the
    recorded trials.csv instead). Beliefs come from belief_delta_rule over completed trials in order."""
    rows = []
    rich_t1 = rng.random() < 0.5
    for b in range(n_blocks):
        hi, lo = draw_block_means(rng, k)
        m = {1: hi if rich_t1 else lo, 2: lo if rich_t1 else hi}
        # 50 instruction rows: 25 T1 + 25 T2, random side, random salience
        ids = rng.permutation(np.r_[np.ones(25), 2 * np.ones(25)])
        for idn in ids:
            right = rng.random() < 0.5
            s = float(rng.random() < 0.5)
            r = float(trial_reward(rng, m[int(idn)], k))
            rows.append(dict(block=b, trial_type=0, id_L=0 if right else idn, id_R=idn if right else 0,
                             sal_L=0 if right else s, sal_R=s if right else 0,
                             rew_L=0 if right else r, rew_R=r if right else 0, mean1=m[1], mean2=m[2]))
        nc = 4 * rng.integers(n_choice[0] // 4, n_choice[1] // 4 + 1)
        conds = [(c, t1l) for c in (0, 1) for t1l in (0, 1)] * (nc // 4)
        rng.shuffle(conds)
        for conflict, t1l in conds:
            idL, idR = (1, 2) if t1l else (2, 1)
            rich_id = 1 if m[1] >= m[2] else 2
            rich_left = idL == rich_id
            sal_hi_left = rich_left if not conflict else not rich_left
            rL = float(trial_reward(rng, m[idL], k))
            rR = float(trial_reward(rng, m[idR], k))
            rows.append(dict(block=b, trial_type=1, id_L=idL, id_R=idR, sal_L=float(sal_hi_left),
                             sal_R=float(not sal_hi_left), rew_L=rL, rew_R=rR, mean1=m[1], mean2=m[2],
                             conflict=conflict))
        rich_t1 = not rich_t1
    table = {key: np.array([r.get(key, np.nan) for r in rows], float) for key in
             ["block", "trial_type", "id_L", "id_R", "sal_L", "sal_R", "rew_L", "rew_R", "mean1", "mean2", "conflict"]}
    n = len(rows)
    table["fix_dur"] = np.round(rng.uniform(cfg.fix_min, cfg.fix_max, n) / cfg.dt) * cfg.dt
    table["delay"] = np.full(n, cfg.delay)
    table["onset_L"] = np.zeros(n)
    table["onset_R"] = np.zeros(n)
    table["side_bias"] = np.zeros(n)
    table["engaged"] = np.ones(n)
    table["trial"] = np.arange(n, dtype=float)
    V1, V2 = belief_delta_rule(table, alpha=alpha, v0=v0, noise=belief_noise, rng=rng)
    table["V1"], table["V2"] = V1, V2
    return table


def belief_delta_rule(table, alpha=0.1, v0=None, noise=0.0, rng=None, chosen_id=None, completed=None):
    """Per-trial belief [V1, V2] (ms) BEFORE each trial, updated after completed trials with the reward of the
    identity received. Single-target trials: the shown identity. Choice trials: `chosen_id` if given (from the
    animal or a simulation), else the richer identity is assumed chosen. `completed` (bool array) skips failures."""
    n = len(table["trial_type"])
    v = np.array([v0 if v0 is not None else 114.0] * 2, float)
    V = np.zeros((n, 2))
    rng = rng or np.random.default_rng(0)
    for i in range(n):
        V[i] = v + (rng.normal(0, noise, 2) if noise > 0 else 0)
        if completed is not None and not completed[i]:
            continue
        if table["trial_type"][i] == 0:
            idn = table["id_L"][i] if table["id_L"][i] > 0 else table["id_R"][i]
            r = table["rew_L"][i] if table["id_L"][i] > 0 else table["rew_R"][i]
        else:
            if chosen_id is not None:
                idn = chosen_id[i]
            else:
                idn = 1 if v[0] >= v[1] else 2
            r = table["rew_L"][i] if table["id_L"][i] == idn else table["rew_R"][i]
        if idn in (1, 2):
            j = int(idn) - 1
            v[j] += alpha * (r - v[j])
    return np.clip(V[:, 0], 0, 400), np.clip(V[:, 1], 0, 400)


# ----------------------------------------------------------------------------- inputs
def n_steps(cfg, table):
    end = table["fix_dur"] + np.maximum(table["delay"], 0) + np.maximum(table["onset_L"], table["onset_R"]) \
        + np.maximum(0, -table["delay"])
    t_go = table["fix_dur"] + table["delay"]
    return int(np.ceil((max(end.max(), (t_go + cfg.response_window).max()) + 30) / cfg.dt)) + 1


def build_inputs(cfg, table, device, generator=None):
    """Return dict of input tensors [B, T, ch] plus timing tensors (ms) and time axis."""
    n_rows = len(table["trial_type"])
    table = {**{c: np.full(n_rows, v, float) for c, v in DEFAULTS.items() if c not in table}, **table}
    tb = {k: torch.as_tensor(np.asarray(v, float), dtype=torch.float32, device=device) for k, v in table.items()
          if k in COLS}
    B = tb["fix_dur"].shape[0]
    T = n_steps(cfg, table)
    t = torch.arange(T, device=device, dtype=torch.float32) * cfg.dt
    t_go = tb["fix_dur"] + tb["delay"]
    on = torch.stack([tb["fix_dur"] + tb["onset_L"], tb["fix_dur"] + tb["onset_R"]], 1)    # [B, 2]
    ids = torch.stack([tb["id_L"], tb["id_R"]], 1)
    sal = torch.stack([tb["sal_L"], tb["sal_R"]], 1)
    present = (ids > 0).float()

    fix = (t[None, :] < (t_go + cfg.go_latency)[:, None]).float()[..., None]                # [B,T,1]
    tv = t[None, :, None] - (on + cfg.vis_latency)[:, None, :]                                # [B,T,2]
    after = (tv >= 0).float()
    vis = after * present[:, None, :] * (1 + cfg.sal_gain * sal[:, None, :]) * \
        (cfg.vis_sustained + cfg.vis_transient * torch.exp(-tv.clamp(min=0) / cfg.vis_tau))
    # identity features per location: [T1@L, T2@L, T1@R, T2@R]
    tu = t[None, :, None] - (on + cfg.up_latency)[:, None, :]
    id_sc, id_up = [], []
    for loc in (0, 1):
        for k in (1, 2):
            m = (ids[:, loc] == k).float()[:, None]
            id_sc.append(after[:, :, loc] * m)
            id_up.append((tu[:, :, loc] >= 0).float() * m)
    id_sc = torch.stack(id_sc, -1)
    id_up = torch.stack(id_up, -1)
    Vb = torch.stack([tb["V1"], tb["V2"]], 1)
    eng = tb["engaged"][:, None]
    Vb = eng * Vb + (1 - eng) * Vb.mean(1, keepdim=True)             # mixture route: disengaged -> no value difference
    belief = Vb[:, None, :].expand(B, T, 2) / cfg.value_scale
    bias = torch.stack([-tb["side_bias"] / 2, tb["side_bias"] / 2], 1)[:, None, :].expand(B, T, 2)
    # slow OU noise onto the two pools (shared + independent components)
    slow = torch.zeros(B, T, 2, device=device)
    if cfg.slow_shared > 0 or cfg.slow_side > 0:
        a = cfg.dt / cfg.slow_tau
        z_sh = torch.zeros(B, device=device)
        z_si = torch.zeros(B, 2, device=device)
        s = np.sqrt(2 * a)
        for i in range(T):
            z_sh = (1 - a) * z_sh + s * torch.randn(B, device=device, generator=generator)
            z_si = (1 - a) * z_si + s * torch.randn(B, 2, device=device, generator=generator)
            slow[:, i] = cfg.slow_shared * z_sh[:, None] + cfg.slow_side * z_si
    return dict(fix=fix, vis=vis, id_sc=id_sc, id_up=id_up, belief=belief, bias=bias, slow=slow,
                t=t, t_go=t_go, on=on, present=present, ids=ids,
                rew=torch.stack([tb["rew_L"], tb["rew_R"]], 1) / cfg.value_scale,
                idsign=torch.where(ids == 1, 1.0, torch.where(ids == 2, -1.0, 0.0)))
