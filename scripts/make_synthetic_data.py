#!/usr/bin/env python3
"""
Synthetic sessions for the recovery test (specs/CONTEXT_INPUTS_PLAN.md, Stage D).

A teacher network (a model fitted with scripts/train_context.py) is run on a real
session's trials (task, context and drift inputs, with recurrent noise); spike
counts for the recorded units are drawn from Poisson distributions with the
teacher's rates and written in the export format, so students can be fit with
train_context.py --data <file> exactly as on real data. The file holds the
teacher's trial set (e.g. verified trials only), so students use --trials all.

Teacher manipulations:
    --ctx-scale g        multiply the teacher's context input weights by g
                         (sweeps the size of the context effects)
    --ie-transform none | rowperm
                         rowperm: permute the E targets of the I->E block with one
                         permutation shared by all I units, so each E unit receives
                         another E unit's full inhibitory input (inhibition
                         statistics preserved, alignment with E-unit identity
                         destroyed)
    --drift-amp a        multiply each unit's rates by the recorded neuron's own slow rate
                         fluctuation (pre-target to visual window, Gaussian-smoothed over
                         --drift-sigma trials, divided by its mean) raised to the power a (renormalized to mean 1) and
                         circularly shifted by about a quarter of the session (whole reward
                         cycles plus a quarter cycle), so the synthetic data
                         carry realistic, context-unrelated slow fluctuations on top of the
                         teacher's own (a = 0: none)

Usage:
    python scripts/make_synthetic_data.py --teacher results/context_pilot/sharedI_free/08_15_seed1 \
        --ctx-scale 2 --out data/synthetic/T_local_08_15_g2.mat
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.sessions import load_export
from src.analysis.pipeline_selectivity import get_target_onset_bins, compute_window_rates
from evaluate_context_models import load_context_model, build_inputs

DRIFT_WINDOW_MS = (-300.0, 250.0)
REWARD_CYCLE = 96  # trials; the reward map alternates every 48 trials in every session


def slow_gain(data: dict, sigma: float, amp: float) -> np.ndarray:
    """[trials, neurons] multiplicative slow fluctuation from the recordings, shifted out of alignment."""
    rates = np.transpose(data['firing_rates'], (2, 1, 0))
    window = compute_window_rates(rates, get_target_onset_bins(data), window_ms=DRIFT_WINDOW_MS)
    n = window.shape[0]
    t = np.arange(n)
    kernel = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma) ** 2)  # [trials, trials]
    ok = ~np.isnan(window)
    smooth = (kernel @ np.where(ok, window, 0)) / np.maximum(kernel @ ok, 1e-9)
    gain = smooth / np.nanmean(window, axis=0, keepdims=True).clip(0.5)
    gain = np.clip(np.nan_to_num(gain, nan=1.0), 0.2, 5.0) ** amp
    gain = np.clip(gain / gain.mean(axis=0, keepdims=True), 0.1, 10.0)  # mean rate preserved
    # Whole reward cycles (2 x 48-trial blocks) plus a quarter cycle: switch-locked parts of the
    # recorded fluctuation then contribute nothing to the switch-triggered contrast (as for
    # train_context.py --context-shift-trials 24)
    return np.roll(gain, REWARD_CYCLE * max(1, round(n / 4 / REWARD_CYCLE)) + REWARD_CYCLE // 4, axis=0)


def synthesize(teacher_dir: Path, ctx_scale: float = 1.0, ie_transform: str = 'none', drift_amp: float = 0.0,
               drift_sigma: float = 15.0, seed: int = 0) -> tuple:
    """(synthetic export dict, real export dict, teacher config) for one teacher."""
    model, config = load_context_model(teacher_dir)
    data = load_export(config['session'], config.get('trials', 'all'), path=config['data_path'])
    inputs = build_inputs(data, config)

    with torch.no_grad():
        if model.n_context > 0 and ctx_scale != 1.0:
            for name in ('W_ctx_e', 'W_ctx_i', 'W_ctx_free'):
                if hasattr(model, name):
                    getattr(model, name).mul_(ctx_scale)
        if ie_transform == 'rowperm':
            assert model.ie_mode == 'free', "rowperm expects a free I->E teacher"
            perm = torch.from_numpy(np.random.default_rng(seed).permutation(model.n_exc))
            n_exc = model.n_exc
            block = model.W_rec_raw.data[:n_exc, n_exc:].clone()
            model.W_rec_raw.data[:n_exc, n_exc:] = block[perm]
        torch.manual_seed(seed)
        rates, _ = model(inputs)

    rec_units = np.array(config['recorded_unit_indices'])
    rates = rates.numpy()[:, :, rec_units]  # [trials, bins, neurons]
    if drift_amp > 0:
        rates = rates * slow_gain(data, drift_sigma, drift_amp)[:, None, :]
    bin_s = float(data['bin_size_ms']) / 1000.0
    counts = np.random.default_rng(seed + 1).poisson(np.clip(rates, 0, None) * bin_s)

    synthetic = {k: v for k, v in data.items() if not k.startswith('_')}
    synthetic['firing_rates'] = np.transpose(counts / bin_s, (2, 1, 0))  # [neurons, bins, trials]
    synthetic['session_name'] = f"synthetic_{config['session']}"
    return synthetic, data, config


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic session from a teacher network')
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--ctx-scale', type=float, default=1.0)
    parser.add_argument('--ie-transform', default='none', choices=['none', 'rowperm'])
    parser.add_argument('--drift-amp', type=float, default=0.0)
    parser.add_argument('--drift-sigma', type=float, default=15.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    teacher_dir = Path(args.teacher)
    synthetic, data, config = synthesize(teacher_dir, args.ctx_scale, args.ie_transform, args.drift_amp,
                                         args.drift_sigma, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(out, synthetic, do_compression=True)
    meta = {'teacher': str(teacher_dir), 'ctx_scale': args.ctx_scale, 'ie_transform': args.ie_transform,
            'drift_amp': args.drift_amp, 'drift_sigma': args.drift_sigma,
            'seed': args.seed, 'teacher_config': {k: config[k] for k in
                                                   ('session', 'context', 'context_targets', 'ie_mode', 'seed', 'trials')}}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}: mean rate {synthetic['firing_rates'].mean():.1f} sp/s "
          f"(real {np.nanmean(data['firing_rates']):.1f})")


if __name__ == '__main__':
    main()
