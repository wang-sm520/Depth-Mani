"""Depth, native joint state, and language policy for recorded Airbot actions."""

import torch
from torch import nn

from depth_policy.model import ImageEncoder, masked_action_loss


CHECKPOINT_FORMAT = "airbot_depth_absolute_joint_policy"
IMAGE_INPUT_PRECISION = "float16_then_float32"


class AirbotPolicyModel(nn.Module):
    """Predict normalized absolute action chunks; dimensions are dataset metadata."""

    def __init__(self, vocabulary_size, state_dim=7, action_dim=7, views=2, horizon=8):
        super().__init__()
        dimensions = {"vocabulary_size": vocabulary_size, "state_dim": state_dim,
                      "action_dim": action_dim, "views": views, "horizon": horizon}
        if any(type(value) is not int or value < 1 for value in dimensions.values()):
            raise ValueError("Policy dimensions must be positive integers")
        if vocabulary_size < 2:
            raise ValueError("Vocabulary must include padding and unknown tokens")
        self.config = dimensions
        self.state_dim, self.action_dim = state_dim, action_dim
        self.views, self.horizon = views, horizon
        self.embedding = nn.Embedding(vocabulary_size, 64, padding_idx=0)
        self.language = nn.GRU(64, 128, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
        )
        self.image_encoder = ImageEncoder(2)
        self.fusion = nn.Sequential(
            nn.Linear(4096 * views + 256, 512), nn.SiLU(),
            nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, 512), nn.SiLU(),
        )
        self.action_head = nn.Linear(512, horizon * action_dim)

    def forward(self, state, tokens, images):
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"Expected state [batch,{self.state_dim}]")
        batch = state.shape[0]
        if tokens.ndim != 2 or tokens.shape[0] != batch or tokens.shape[1] < 1:
            raise ValueError("Expected language tokens [batch,length]")
        if (images.ndim != 5 or images.shape[:3] != (batch, self.views, 2)
                or min(images.shape[-2:]) < 16):
            raise ValueError(f"Expected depth [batch,{self.views},2,height,width], size >=16")
        lengths = (tokens != 0).sum(dim=1).clamp_min(1)
        language, _ = self.language(self.embedding(tokens))
        language = language[torch.arange(batch, device=tokens.device), lengths - 1]
        # The training dataset retains depth in half precision in host RAM.
        # Apply the identical rounding to online float32 depth before encoding.
        images = images.to(dtype=torch.float16).float()
        encoded = self.image_encoder(images.reshape(batch * self.views, 2, *images.shape[-2:]))
        features = torch.cat([encoded.reshape(batch, -1), self.state_encoder(state), language], dim=1)
        return self.action_head(self.fusion(features)).reshape(batch, self.horizon, self.action_dim)


def batch_loss(model, batch):
    prediction = model(batch["state"], batch["tokens"], batch["images"])
    if not torch.isfinite(prediction).all() or not torch.isfinite(batch["action"]).all():
        raise FloatingPointError("Nonfinite policy prediction or action label")
    loss = masked_action_loss(prediction, batch["action"], batch["mask"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite action loss")
    return loss
