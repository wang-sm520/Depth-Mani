import torch
from torch import nn
from torch.nn import functional as functional


class ImageEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        blocks = []
        for width in (32, 64, 128, 256):
            blocks.extend([
                nn.Conv2d(channels, width, 3, stride=2, padding=1),
                nn.GroupNorm(8, width), nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            ])
            channels = width
        self.layers = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())

    def forward(self, images):
        return self.layers(images)


class StudentPolicy(nn.Module):
    def __init__(self, modality, vocabulary_size, horizon=8):
        super().__init__()
        self.modality = modality
        self.horizon = horizon
        self.embedding = nn.Embedding(vocabulary_size, 64, padding_idx=0)
        self.language = nn.GRU(64, 128, batch_first=True)
        self.state_encoder = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        if modality == "state":
            self.image_encoder = None
            self.fusion = nn.Sequential(
                nn.Linear(256, 2048), nn.SiLU(), nn.Linear(2048, 2048), nn.SiLU(),
                nn.Linear(2048, 512), nn.SiLU(),
            )
        else:
            self.image_encoder = ImageEncoder(2 if modality == "depth" else 3)
            self.fusion = nn.Sequential(
                nn.Linear(8192 + 256, 512), nn.SiLU(),
                nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, 512), nn.SiLU(),
            )
        self.action_head = nn.Linear(512, horizon * 7)

    def forward(self, state, tokens, images=None):
        lengths = (tokens != 0).sum(dim=1).clamp_min(1)
        language, _ = self.language(self.embedding(tokens))
        language = language[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        features = [self.state_encoder(state), language]
        if self.image_encoder is not None:
            if images is None:
                raise ValueError("Visual policy requires images")
            batch, views, channels, height, width = images.shape
            if views != 2:
                raise ValueError("Expected two camera views")
            encoded = self.image_encoder(images.float().reshape(batch * views, channels, height, width))
            features.insert(0, encoded.reshape(batch, -1))
        output = self.action_head(self.fusion(torch.cat(features, dim=1)))
        return output.reshape(-1, self.horizon, 7)


def masked_action_loss(prediction, target, mask):
    if not mask.any():
        raise ValueError("No valid action labels")
    errors = functional.smooth_l1_loss(prediction, target, reduction="none")
    return (errors * mask[..., None]).sum() / (mask.sum() * target.shape[-1])
