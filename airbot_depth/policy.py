"""Offline Airbot action inference; this module never connects to a robot."""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path

import numpy as np
import torch

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.depth import DepthAnythingTransform, canonical_depth_config
from airbot_depth.model import AirbotPolicyModel, CHECKPOINT_FORMAT, IMAGE_INPUT_PRECISION
from depth_policy.data import build_vocabulary, encode_language


class AirbotDepthPolicy:
    """Load a trusted local training checkpoint and return absolute action chunks.

    State/action order and units are copied from the original dataset metadata.
    RGB and precomputed depth calls share the saved preprocessing configuration.
    Only the instruction present in the training data is supported.
    """

    def __init__(self, checkpoint, device="cpu", local_files_only=False):
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Unsupported Airbot policy checkpoint format")
        if payload.get("image_input_precision") != IMAGE_INPUT_PRECISION:
            raise ValueError("Checkpoint image precision differs from training/inference implementation")
        self.manifest = payload["manifest"]
        manifest = self.manifest
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ValueError("Checkpoint was not trained on a completed dataset")
        if (payload.get("action_semantics") != "absolute_joint_position"
                or manifest.get("action_semantics") != payload["action_semantics"]):
            raise ValueError("Checkpoint must predict native absolute joint positions")
        self.camera_keys = list(manifest["camera_keys"])
        if not self.camera_keys or len(self.camera_keys) != len(set(self.camera_keys)):
            raise ValueError("Checkpoint has invalid camera ordering")
        self.prompt = manifest["prompt"]
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Checkpoint needs a nonempty trained instruction")
        self.vocabulary = payload["vocabulary"]
        if self.vocabulary != build_vocabulary([{"instruction": self.prompt}]):
            raise ValueError("Checkpoint vocabulary does not match the trained prompt")
        self.tokens = encode_language(self.prompt, self.vocabulary)
        self.depth_config = payload["depth_config"]
        self.depth_provenance = payload["depth_provenance"]
        if (not self.depth_config or not self.depth_provenance
                or self.depth_config != manifest["depth_config"]
                or self.depth_provenance != manifest["depth_provenance"]):
            raise ValueError("Checkpoint depth preprocessing differs from its training manifest")
        if (canonical_depth_config(self.depth_config) != self.depth_config
                or self.depth_config["image_size"] < 16):
            raise ValueError("Checkpoint depth preprocessing must be canonical with image_size >=16")
        config = payload["model_config"]
        expected = {"vocabulary_size": len(self.vocabulary), "state_dim": manifest["state_dim"],
                    "action_dim": manifest["action_dim"], "views": len(self.camera_keys),
                    "horizon": payload["config"]["horizon"]}
        if config != expected:
            raise ValueError("Checkpoint model dimensions differ from dataset/config metadata")
        for kind in ("state", "action"):
            if len(manifest.get(f"{kind}_names", [])) != config[f"{kind}_dim"]:
                raise ValueError(f"Checkpoint lacks the recorded {kind} ordering")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Airbot depth policy supports cpu or cuda devices")
        if self.device.type == "cuda":
            workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", self.depth_config["cublas_workspace_config"])
            if workspace != self.depth_config["cublas_workspace_config"]:
                raise ValueError("Depth inference requires the recorded CUBLAS_WORKSPACE_CONFIG")
        self.model = AirbotPolicyModel(**config)
        self.model.load_state_dict(payload["model"], strict=True)
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Checkpoint contains nonfinite model weights")
        self.model.to(self.device).eval()
        self.statistics = {}
        for kind in ("state", "action"):
            for statistic in ("mean", "std"):
                key = f"{kind}_{statistic}"
                values = np.asarray(payload["statistics"][key], dtype=np.float32)
                if values.shape != (config[f"{kind}_dim"],) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid checkpoint statistic: {key}")
                if statistic == "std" and np.any(values <= 0):
                    raise ValueError(f"Checkpoint standard deviations must be positive: {key}")
                self.statistics[key] = values
        self.fps = float(manifest["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid checkpoint dataset frame rate")
        self.local_files_only = local_files_only
        self._depth_transform = None

    def _validate_state_and_prompt(self, state, prompt):
        if prompt is not None and prompt != self.prompt:
            raise ValueError("This checkpoint supports only its exact trained prompt")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.model.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite state [{self.model.state_dim}] in recorded joint order")
        normalized = (state - self.statistics["state_mean"]) / self.statistics["state_std"]
        if not np.isfinite(normalized).all():
            raise ValueError("State normalization produced nonfinite values")
        return normalized

    @torch.inference_mode()
    def infer_depth(self, state, depth, prompt=None):
        """Infer from the saved normalized depth+validity representation, not meters."""
        normalized = self._validate_state_and_prompt(state, prompt)
        depth = np.asarray(depth, dtype=np.float32)
        size = self.depth_config["image_size"]
        expected = (len(self.camera_keys), 2, size, size)
        if depth.shape != expected or not np.isfinite(depth).all():
            raise ValueError(f"Expected finite preprocessed depth {expected}")
        if np.any(depth < 0) or np.any(depth > 1):
            raise ValueError("Preprocessed relative depth and validity must be in [0,1]")
        if not np.isin(depth[:, 1], [0.0, 1.0]).all():
            raise ValueError("Depth validity channel must contain only zero or one")
        if np.any(depth[:, 0][depth[:, 1] == 0] != 0):
            raise ValueError("Invalid or padded depth pixels must be zero")
        prediction = self.model(
            torch.from_numpy(normalized[None]).to(self.device),
            torch.from_numpy(self.tokens[None]).to(self.device),
            torch.from_numpy(np.ascontiguousarray(depth[None])).to(self.device),
        )[0].cpu().numpy()
        actions = prediction * self.statistics["action_std"] + self.statistics["action_mean"]
        if actions.shape != (self.model.horizon, self.model.action_dim) or not np.isfinite(actions).all():
            raise FloatingPointError("Policy produced nonfinite or incorrectly shaped absolute actions")
        return actions.astype(np.float32, copy=False)

    def infer_rgb(self, state, rgb, prompt=None):
        """Use exact saved camera order and preprocessing, returning native actions."""
        self._validate_state_and_prompt(state, prompt)
        if not isinstance(rgb, Mapping) or set(rgb) != set(self.camera_keys):
            raise ValueError(f"RGB observation must contain exactly these camera keys: {self.camera_keys}")
        images = []
        for key in self.camera_keys:
            value = np.asarray(rgb[key])
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"Camera {key} must be a uint8 RGB image [height,width,3]")
            images.append(value)
        if self._depth_transform is None:
            transform = DepthAnythingTransform(self.depth_config, device=str(self.device),
                                               local_files_only=self.local_files_only)
            if transform.config != self.depth_config or transform.provenance() != self.depth_provenance:
                raise ValueError("Runtime Depth Anything artifacts/preprocessing differ from the training conversion")
            self._depth_transform = transform
        return self.infer_depth(state, self._depth_transform(images), prompt=prompt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted local Airbot training checkpoint")
    parser.add_argument("--observation", required=True,
                        help="NPZ containing state and either depth or the exact recorded RGB camera keys")
    parser.add_argument("--output", required=True, help="New JSON action output; never sent to a robot")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {output}")
    policy = AirbotDepthPolicy(args.checkpoint, device=args.device, local_files_only=args.local_files_only)
    with np.load(args.observation, allow_pickle=False) as observation:
        if "state" not in observation.files:
            raise ValueError("Observation NPZ must contain state")
        fields = set(observation.files)
        if "depth" in fields:
            if fields != {"state", "depth"}:
                raise ValueError("Depth observation NPZ must contain exactly state and depth")
            actions = policy.infer_depth(observation["state"], observation["depth"])
        else:
            if fields != {"state", *policy.camera_keys}:
                raise ValueError("RGB observation NPZ must contain state and the exact recorded camera keys")
            actions = policy.infer_rgb(observation["state"],
                                       {key: observation[key] for key in policy.camera_keys})
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"schema_version": 1, "checkpoint_sha256": policy.checkpoint_sha256,
                         "prompt": policy.prompt, "camera_keys": policy.camera_keys,
                         "state_names": policy.manifest["state_names"],
                         "action_names": policy.manifest["action_names"],
                         "action_semantics": "absolute_joint_position",
                         "dataset_fps": policy.fps, "actions": actions.tolist(),
                         "robot_executed": False})


if __name__ == "__main__":
    main()
