#!/usr/bin/env python3
"""
Context-effect size of candidate teachers for the recovery test.

For a fitted model and context gains g, the model's mean rates (5 noise draws) are
computed with its context input weights scaled by g, and the per-neuron pre-target
reward-context effect is measured as for the recordings. Reported per gain: the
population mean and SD across neurons of the effect, its r with the recorded effect,
and the part due to context: the SD across neurons of effect(g) - effect(g = 0). A usable
teacher has a context-driven SD comparable to the recorded SD.

Usage:
    python scripts/teacher_effect_size.py results/grid_v12/shared_free/08_14_seed3 --gains 0,1,3,10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.sessions import load_export, stable_mask
from src.analysis.pipeline_selectivity import get_target_onset_bins, compute_window_rates
from evaluate_context_models import load_context_model, build_inputs, run_with_currents, effects, N_DRAWS
import analyze_context_effects as A


def model_effects(model_dir: Path, gain: float, sess: dict, data: dict) -> dict:
    """Per-neuron effects of a model's rates with context weights scaled by gain."""
    model, config = load_context_model(model_dir)
    with torch.no_grad():
        for name in ('W_ctx_e', 'W_ctx_i', 'W_ctx_free'):
            if hasattr(model, name):
                getattr(model, name).mul_(gain)
    inputs = build_inputs(data, config)
    rec = np.array(config['recorded_unit_indices'])
    rates = sum(run_with_currents(model, inputs, s)[0][:, :, rec] for s in range(N_DRAWS)) / N_DRAWS
    rates = np.where(stable_mask(data).T[:, None, :], rates, np.nan)
    onset = get_target_onset_bins(data)
    m = dict(sess)
    m['vis'] = compute_window_rates(rates, onset)
    m['pre'] = compute_window_rates(rates, onset, window_ms=A.PRE_WINDOW_MS)
    return effects(m)


def main():
    parser = argparse.ArgumentParser(description='Teacher context-effect size vs gain')
    parser.add_argument('model_dirs', nargs='+')
    parser.add_argument('--gains', default='0,1,3,10')
    args = parser.parse_args()
    gains = [float(g) for g in args.gains.split(',')]
    if 0.0 not in gains:
        gains = [0.0] + gains

    for d in args.model_dirs:
        d = Path(d)
        _, config = load_context_model(d)
        data = load_export(config['session'], config['trials'])
        sess = A.session_from_data(data, config['session'])
        rec = effects(sess)['reward_pre']
        out = {g: model_effects(d, g, sess, data)['reward_pre'] for g in gains}
        print(f"{d}  recorded reward_pre: mean {np.nanmean(rec):+.2f}, SD {np.nanstd(rec):.2f}")
        for g in gains:
            e = out[g]
            ok = ~np.isnan(e) & ~np.isnan(rec)
            ctx_part = e - out[0.0]
            r_ctx = np.corrcoef(ctx_part[ok], rec[ok])[0, 1] if g > 0 and np.nanstd(ctx_part) > 0 else np.nan
            print(f"  g={g:<4g} mean {np.nanmean(e):+.2f} SD {np.nanstd(e):.2f} | r with recorded "
                  f"{np.corrcoef(e[ok], rec[ok])[0, 1]:+.2f} | context part SD {np.nanstd(ctx_part):.2f}, "
                  f"r with recorded {r_ctx:+.2f}")


if __name__ == '__main__':
    main()
