#!/usr/bin/env python3
"""
Synthetic sessions for the recovery test (specs/CONTEXT_INPUTS_PLAN.md, Stage D).

A teacher network (a model fitted with scripts/train_context.py) is run on a real
session's trials (task, context and drift inputs, with recurrent noise); spike
counts for the recorded units are drawn from Poisson distributions with the
teacher's rates and written in the export format, so students can be fit with
train_context.py --data <file> exactly as on real data.

Teacher manipulations:
    --ctx-scale g        multiply the teacher's context input weights by g
                         (sweeps the size of the context effects)
    --ie-transform none | rowperm
                         rowperm: permute the E targets of the I->E block with one
                         permutation shared by all I units, so each E unit receives
                         another E unit's full inhibitory input (inhibition
                         statistics preserved, alignment with E-unit identity
                         destroyed)

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

from src.data_loader import load_mat_file
from evaluate_context_models import load_context_model, build_inputs


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic session from a teacher network')
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--ctx-scale', type=float, default=1.0)
    parser.add_argument('--ie-transform', default='none', choices=['none', 'rowperm'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    teacher_dir = Path(args.teacher)
    model, config = load_context_model(teacher_dir)
    data = load_mat_file(config['data_path'])
    inputs = build_inputs(data, config)

    with torch.no_grad():
        if model.n_context > 0 and args.ctx_scale != 1.0:
            for name in ('W_ctx_e', 'W_ctx_i', 'W_ctx_free'):
                if hasattr(model, name):
                    getattr(model, name).mul_(args.ctx_scale)
        if args.ie_transform == 'rowperm':
            assert model.ie_mode == 'free', "rowperm expects a free I->E teacher"
            perm = torch.from_numpy(np.random.default_rng(args.seed).permutation(model.n_exc))
            n_exc = model.n_exc
            block = model.W_rec_raw.data[:n_exc, n_exc:].clone()
            model.W_rec_raw.data[:n_exc, n_exc:] = block[perm]
        torch.manual_seed(args.seed)
        rates, _ = model(inputs)

    rec_units = np.array(config['recorded_unit_indices'])
    rates = rates.numpy()[:, :, rec_units]  # [trials, bins, neurons]
    bin_s = float(data['bin_size_ms']) / 1000.0
    counts = np.random.default_rng(args.seed + 1).poisson(np.clip(rates, 0, None) * bin_s)

    synthetic = {k: v for k, v in data.items() if not k.startswith('_')}
    synthetic['firing_rates'] = np.transpose(counts / bin_s, (2, 1, 0))  # [neurons, bins, trials]
    synthetic['session_name'] = f"synthetic_{config['session']}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(out, synthetic, do_compression=True)
    meta = {'teacher': str(teacher_dir), 'ctx_scale': args.ctx_scale, 'ie_transform': args.ie_transform,
            'seed': args.seed, 'teacher_config': {k: config[k] for k in
                                                   ('session', 'context', 'context_targets', 'ie_mode', 'seed')}}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}: mean rate {synthetic['firing_rates'].mean():.1f} sp/s "
          f"(real {np.nanmean(data['firing_rates']):.1f})")


if __name__ == '__main__':
    main()
