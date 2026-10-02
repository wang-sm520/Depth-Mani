"""Frozen BC for the RL env: batched action chunks plus the fusion feature the residual actor sees.

Inputs match deployment: head = Depth Anything V2 on the head RGB (the checkpoint's pinned transform),
wrist = D405 metric depth through airbot_rl.depth.
"""

import numpy as np
import torch
import torch.nn.functional as F

from airbot_depth.depth import DepthAnythingTransform, _deterministic_float32, normalize_relative_depth
from airbot_depth.policy import AirbotDepthPolicy
from airbot_rl.depth import encode_wrist_depth


def head_depth(transform, rgb, batch=32):
    """uint8 [N,H,W,3] -> [N,2,128,128]: DA2 in batches (activation memory), same result as transform's per-image call."""
    config = transform.config
    if not hasattr(transform, "_gpu_input_size"):
        sample = transform._processor(
            images=[np.ascontiguousarray(rgb[0].cpu().numpy())],
            input_data_format="channels_last", data_format="channels_first",
            return_tensors="pt", **transform._processor_settings)["pixel_values"]
        transform._gpu_input_size = sample.shape[-2:]
        transform._gpu_mean = torch.tensor(transform._processor_settings["image_mean"],
                                           device=transform._device).view(1, 3, 1, 1)
        transform._gpu_std = torch.tensor(transform._processor_settings["image_std"],
                                          device=transform._device).view(1, 3, 1, 1)
    rgb = rgb.to(transform._device)
    prediction = []
    with _deterministic_float32(transform._device.type):
        for i in range(0, len(rgb), batch):
            pixel_values = rgb[i:i + batch].permute(0, 3, 1, 2).float()
            pixel_values = F.interpolate(pixel_values, size=transform._gpu_input_size,
                                         mode="bicubic", align_corners=False)
            pixel_values = pixel_values.clamp(0, 255).round().div_(255)
            pixel_values = (pixel_values - transform._gpu_mean) / transform._gpu_std
            with torch.autocast(device_type=transform._device.type, dtype=torch.float16,
                                enabled=transform._device.type == "cuda"):
                prediction.append(transform._model(pixel_values=pixel_values).predicted_depth.float().cpu().numpy())
    prediction = np.concatenate(prediction)
    return torch.from_numpy(np.stack([normalize_relative_depth(
        p, config["percentiles"], config["image_size"], source_shape=tuple(rgb.shape[1:3])) for p in prediction]))


class FrozenBC:
    def __init__(self, checkpoint, device):
        self.policy = AirbotDepthPolicy(checkpoint, device=device)
        self.model = self.policy.model.requires_grad_(False).eval()
        self.depth = DepthAnythingTransform(self.policy.depth_config, device=device)
        if self.depth.provenance() != self.policy.depth_provenance:
            raise ValueError("Runtime Depth Anything differs from the checkpoint's training conversion")
        stats = {key: torch.as_tensor(value, device=device) for key, value in self.policy.statistics.items()}
        self.state_mean, self.state_std = stats["state_mean"], stats["state_std"]
        self.action_mean, self.action_std = stats["action_mean"], stats["action_std"]
        self.tokens = torch.as_tensor(self.policy.tokens, device=device)[None]
        self.feature_dim = self.model.action_head.in_features
        self._features = None
        self.model.action_head.register_forward_pre_hook(lambda _, inputs: setattr(self, "_features", inputs[0]))

    @torch.no_grad()
    def __call__(self, state, head_rgb, wrist_metres):
        """state [N,7] native, head uint8 [N,H,W,3], wrist [N,h,w] m -> (chunk [N,8,7] native, features, norm state)."""
        self.last_depth = torch.stack((head_depth(self.depth, head_rgb).to(state.device), encode_wrist_depth(wrist_metres)), 1)
        state = (state - self.state_mean) / self.state_std
        chunk = self.model(state, self.tokens.expand(len(state), -1), self.last_depth)
        return chunk * self.action_std + self.action_mean, self._features, state
