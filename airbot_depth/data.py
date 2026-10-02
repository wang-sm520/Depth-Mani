import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from airbot_depth.common import sha256_file
from depth_policy.data import encode_language


def load_manifest(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Conversion must be complete before training")
    if manifest["action_semantics"] != "absolute_joint_position":
        raise ValueError("Unsupported action semantics")
    episodes = manifest["episodes"]
    ids = [item["episode_index"] for item in episodes]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(manifest["source"]["selected_episode_indices"]):
        raise ValueError("Duplicate or incomplete converted episodes")
    if any(item["split"] not in {"train", "validation"} for item in episodes):
        raise ValueError("Unknown episode split")
    for episode in episodes:
        if episode.get("instruction") != manifest["prompt"]:
            raise ValueError("Episode instruction differs from the fixed dataset prompt")
        if episode["frames"] < 1:
            raise ValueError("Empty converted episode")
        path = root / episode["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Episode path escapes the converted dataset")
        if sha256_file(path) != episode["sha256"]:
            raise ValueError(f"Converted episode fingerprint mismatch: {path}")
        with h5py.File(path, "r") as stream:
            expected = (episode["frames"], len(manifest["camera_keys"]), 2,
                        manifest["depth_config"]["image_size"], manifest["depth_config"]["image_size"])
            if stream["depth"].shape != expected:
                raise ValueError(f"Depth geometry mismatch: {path}")
            if stream["state"].shape != (episode["frames"], manifest["state_dim"]):
                raise ValueError("State dimension mismatch")
            if stream["action"].shape != (episode["frames"], manifest["action_dim"]):
                raise ValueError("Action dimension mismatch")
            if stream.attrs["episode_index"] != episode["episode_index"] or stream.attrs["instruction"] != manifest["prompt"]:
                raise ValueError("Episode identity/prompt mismatch")
    if not select_episodes(manifest, "train") or not select_episodes(manifest, "validation"):
        raise ValueError("Require nonempty, disjoint episode-level train and validation splits")
    return manifest


def select_episodes(manifest, split):
    return sorted([item for item in manifest["episodes"] if item["split"] == split],
                  key=lambda item: item["episode_index"])


def fit_statistics(data_dir, episodes):
    if not episodes or any(item["split"] != "train" for item in episodes):
        raise ValueError("Statistics must be fit on training episodes only")
    arrays = {"state": [], "action": []}
    for episode in episodes:
        with h5py.File(Path(data_dir) / episode["path"], "r") as stream:
            for name in arrays:
                arrays[name].append(stream[name][:].astype(np.float64))
    result = {}
    for name, chunks in arrays.items():
        values = np.concatenate(chunks)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} statistics")
        result[f"{name}_mean"] = values.mean(axis=0).tolist()
        result[f"{name}_std"] = values.std(axis=0).clip(1e-6).tolist()
    return result


class AirbotDataset(Dataset):
    """Episode-bounded chunks, retaining compressed-to-half depth in host RAM."""

    def __init__(self, data_dir, episodes, statistics, vocabulary, horizon=8):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.records = []
        self.offsets = [0]
        for episode in episodes:
            with h5py.File(Path(data_dir) / episode["path"], "r") as stream:
                state, action = stream["state"][:], stream["action"][:]
                depth = stream["depth"][:]
                if not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)):
                    raise ValueError("Depth and mask must be finite in [0,1]")
                if not np.isin(depth[:, :, 1], [0.0, 1.0]).all():
                    raise ValueError("Depth validity mask must be binary")
                if np.any(depth[:, :, 0][depth[:, :, 1] == 0] != 0):
                    raise ValueError("Invalid or padded depth pixels must be zero")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError("Nonfinite state/action")
                # Both cached and online inputs round to half in the policy before
                # encoding; the on-disk depth is retained at full float32 precision.
                images = torch.from_numpy(depth).half()
            normalized = {}
            for name, array in [("state", state), ("action", action)]:
                std = np.asarray(statistics[f"{name}_std"], dtype=np.float32)
                if np.any(std <= 0) or not np.isfinite(std).all():
                    raise ValueError("Normalization standard deviations must be positive")
                normalized[name] = torch.from_numpy(
                    ((array - np.asarray(statistics[f"{name}_mean"], dtype=np.float32)) / std).astype(np.float32)
                )
            self.records.append({**normalized, "images": images,
                                 "tokens": torch.from_numpy(encode_language(episode["instruction"], vocabulary))})
            self.offsets.append(self.offsets[-1] + len(state))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.offsets, index, side="right") - 1)
        record = self.records[episode]
        frame = index - self.offsets[episode]
        count = min(self.horizon, len(record["state"]) - frame)
        actions = torch.zeros(self.horizon, record["action"].shape[-1], dtype=torch.float32)
        actions[:count] = record["action"][frame:frame + count]
        return {"state": record["state"][frame], "action": actions,
                "images": record["images"][frame], "tokens": record["tokens"],
                "mask": torch.arange(self.horizon) < count}
