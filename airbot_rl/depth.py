"""Wrist depth encoding shared by simulation, BC training and deployment (D405 depth stream, metres).

Same [depth, validity] x 128 x 128 letterbox as the head's DA2 input. Fixed metric range instead of per-frame
percentiles, near = bright like DA2. A resized pixel is valid when at least half its support is; its value
averages the valid support, so isolated D405 holes do not blank whole regions.
"""

import torch
import torch.nn.functional as F

NEAR, FAR = 0.07, 1.0  # m: D405 minimum range; beyond 1 m only floor/background is visible
SIZE = 128


def encode_wrist_depth(depth):
    """[N,H,W] metres (0 or non-finite = missing) -> [N,2,SIZE,SIZE] float32."""
    valid = torch.isfinite(depth) & (depth >= NEAR) & (depth <= FAR)
    value = torch.where(valid, (FAR - depth) / (FAR - NEAR), 0.0)
    h, w = depth.shape[1:]
    oh, ow = round(h * SIZE / max(h, w)), round(w * SIZE / max(h, w))
    small = F.interpolate(torch.stack((value, valid.float()), 1), (oh, ow), mode="bilinear",
                          align_corners=False, antialias=True)
    support = small[:, 1]
    out = torch.zeros(len(depth), 2, SIZE, SIZE, device=depth.device)
    top, left = (SIZE - oh) // 2, (SIZE - ow) // 2
    keep = support >= 0.5
    out[:, 0, top:top + oh, left:left + ow] = torch.where(keep, small[:, 0] / support.clamp_min(1e-6), 0.0).clamp(0, 1)
    out[:, 1, top:top + oh, left:left + ow] = keep.float()
    return out
