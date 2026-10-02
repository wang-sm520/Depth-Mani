from pathlib import Path
import re

import h5py
import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import Dataset

from depth_policy.common import sha256_file
from depth_policy.episodes import episode_index


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(episodes):
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for word in sorted({word for episode in episodes for word in words(episode["instruction"])}):
        vocabulary[word] = len(vocabulary)
    return vocabulary


def encode_language(text, vocabulary, length=24):
    tokens = [vocabulary.get(word, 1) for word in words(text)]
    if not tokens or len(tokens) > length:
        raise ValueError("Instruction is empty or exceeds configured token length")
    return np.asarray(tokens + [0] * (length - len(tokens)), dtype=np.int64)


def preprocess_images(first, second, modality, size=128, depth_max=3.0):
    if modality == "state":
        return None
    images = np.stack([first, second], axis=1)
    tensor = torch.as_tensor(np.ascontiguousarray(images))
    if modality == "depth":
        if tensor.ndim != 4:
            raise ValueError("Depth batch must have shape [batch,views,height,width]")
        valid = torch.isfinite(tensor) & (tensor > 0)
        metric = torch.nan_to_num(tensor.float(), nan=0, posinf=0, neginf=0)
        metric = metric.clamp(0, depth_max) / depth_max
        tensor = torch.stack([metric, valid.float()], dim=2)
    elif modality == "rgb":
        if tensor.dtype != torch.uint8:
            raise ValueError("RGB must be uint8")
        tensor = tensor.permute(0, 1, 4, 2, 3).float() / 255.0
    else:
        raise ValueError(f"Unknown modality: {modality}")
    batch, views, channels = tensor.shape[:3]
    tensor = tensor.reshape(batch * views, channels, *tensor.shape[-2:])
    if modality == "depth":
        tensor = functional.interpolate(tensor, size=(size, size), mode="nearest")
    else:
        tensor = functional.interpolate(tensor, size=(size, size), mode="bilinear",
                                        align_corners=False, antialias=True)
    return tensor.reshape(batch, views, channels, size, size)


def training_episodes(directory, split, limit=None):
    episodes = [item for item in episode_index(directory)
                if item["success"] and item["split"] == split
                and item["teacher"].get("policy_config") == "pi05_libero"]
    episodes.sort(key=lambda item: (item["seed"], item["initial_state_id"], item["episode_id"]))
    if limit:
        episodes = episodes[:limit]
    if not episodes:
        raise ValueError(f"No successful pi05_libero episodes for {split}")
    return episodes


def fit_statistics(episodes):
    if any(item["split"] != "train" for item in episodes):
        raise ValueError("Normalization can only use training episodes")
    states, actions = [], []
    for episode in episodes:
        with h5py.File(episode["path"], "r") as stream:
            selected = stream["steps/phase"][:] == 1
            states.append(stream["steps/state"][:][selected])
            actions.append(stream["steps/executed_action"][:][selected])
    result = {}
    for name, arrays in [("state", states), ("action", actions)]:
        values = np.concatenate(arrays).astype(np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} training statistics")
        result[f"{name}_mean"] = values.mean(axis=0).astype(np.float32).tolist()
        result[f"{name}_std"] = values.std(axis=0).clip(0.01).astype(np.float32).tolist()
    return result


class EpisodeDataset(Dataset):
    def __init__(self, episodes, modality, vocabulary, statistics, horizon=8, image_size=128, depth_max=3.0):
        self.episodes = episodes
        self.horizon = horizon
        self.statistics = statistics
        self.records = []
        self.offsets = [0]
        for episode in episodes:
            with h5py.File(episode["path"], "r") as stream:
                steps = stream["steps"]
                selected = np.flatnonzero(steps["phase"][:] == 1)
                if not len(selected) or np.any(np.diff(selected) != 1):
                    raise ValueError("Demonstration segment must be nonempty and contiguous")
                state = steps["state"][:][selected]
                action = steps["executed_action"][:][selected]
                images = None
                if modality != "state":
                    batches = []
                    for start in range(0, len(selected), 32):
                        indices = selected[start:start + 32]
                        batches.append(preprocess_images(
                            steps[f"{modality}_agentview"][indices],
                            steps[f"{modality}_wrist"][indices], modality, image_size, depth_max
                        ).half())
                    images = torch.cat(batches)
            state = (state - np.asarray(statistics["state_mean"])) / np.asarray(statistics["state_std"])
            action = (action - np.asarray(statistics["action_mean"])) / np.asarray(statistics["action_std"])
            self.records.append({"state": torch.tensor(state, dtype=torch.float32),
                                 "action": torch.tensor(action, dtype=torch.float32),
                                 "images": images,
                                 "tokens": torch.from_numpy(encode_language(episode["instruction"], vocabulary))})
            self.offsets.append(self.offsets[-1] + len(state))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        record = self.records[episode_index]
        offset = index - self.offsets[episode_index]
        count = min(self.horizon, len(record["state"]) - offset)
        target = torch.zeros(self.horizon, 7)
        target[:count] = record["action"][offset:offset + count]
        mask = torch.arange(self.horizon) < count
        result = {"state": record["state"][offset], "tokens": record["tokens"],
                  "action": target, "mask": mask}
        if record["images"] is not None:
            result["images"] = record["images"][offset]
        return result


def data_fingerprint(episodes):
    return [{"path": str(Path(item["path"]).resolve()), "sha256": sha256_file(item["path"]),
             "episode_id": item["episode_id"], "initial_state_id": item["initial_state_id"]}
            for item in episodes]
