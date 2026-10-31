"""Tolerant loader for conflict-task-RNN srs checkpoints: older checkpoints lack buffers added later (e.g. shape_u),
which the current SRSNet registers deterministically from the config; load with strict=False and report missing keys."""
import torch


def load_net(path, device='cpu'):
    from srs.config import SRSConfig
    from srs.network import SRSNet
    ck = torch.load(path, map_location=device)
    cfg = SRSConfig.from_dict(ck['config'])
    cfg.device = device
    net = SRSNet(cfg).to(device)
    missing, unexpected = net.load_state_dict(ck['state_dict'], strict=False)
    assert not unexpected and all(k in ('shape_u',) for k in missing), (missing, unexpected)
    net.eval()
    return net
