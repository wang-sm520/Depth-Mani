import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/pilot.json"):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(root):
    root = Path(root)
    tracked = sorted(root.rglob("*.py")) if root.name == "depth_policy" else []
    result = {str(path.relative_to(root)): sha256_file(path) for path in tracked}
    if not tracked:
        result["git_head"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        result["git_diff_sha256"] = hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD"]
        )).hexdigest()
    return result


def state_vector(observation):
    quaternion = np.array(observation["robot0_eef_quat"], dtype=np.float64, copy=True)
    scalar = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - scalar * scalar))
    rotation = np.zeros(3) if np.isclose(denominator, 0.0) else (
        quaternion[:3] * 2.0 * np.arccos(scalar) / denominator
    )
    state = np.concatenate([
        observation["robot0_eef_pos"], rotation, observation["robot0_gripper_qpos"]
    ]).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid robot state: {state}")
    return state


def orient_image(image):
    return np.ascontiguousarray(image[::-1, ::-1])


def validate_action(action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid action: {action}")
    return action
