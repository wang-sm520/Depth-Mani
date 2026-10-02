import hashlib
import json
import os
from pathlib import Path
import uuid

import h5py
import numpy as np


IMAGE_FIELDS = ("depth_agentview", "depth_wrist", "rgb_agentview", "rgb_wrist")


def image_digest(image):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest(), dtype=np.uint8)


class EpisodeWriter:
    def __init__(self, directory, metadata, initial_state, xml):
        self.record_images = metadata.get("record_images", True)
        if not self.record_images and not metadata.get("evaluation_only"):
            raise ValueError("Compact recording is restricted to evaluation")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.episode_id = uuid.uuid4().hex
        self.temporary = directory / f"{self.episode_id}.partial.h5"
        self.destination = directory / f"{self.episode_id}.h5"
        self.file = h5py.File(self.temporary, "x")
        self.file.attrs["schema_version"] = 1 if self.record_images else 2
        self.file.attrs["episode_id"] = self.episode_id
        self.file.attrs["metadata"] = json.dumps(metadata, allow_nan=False)
        self.file.create_dataset("initial_sim_state", data=initial_state)
        self.file.create_dataset("model_xml", data=xml, dtype=h5py.string_dtype())
        self.steps = self.file.create_group("steps")
        self.count = 0

    def append(self, observation, action, step_index, sim_timestamp, sim_state, phase, camera_pose):
        if not self.record_images:
            if phase == 1 and "first_policy_observation" not in self.file:
                first = self.file.create_group("first_policy_observation")
                first.attrs["step_index"] = step_index
                for name, value in observation.items():
                    first.create_dataset(name, data=value, compression="lzf", shuffle=True)
            observation = {**{name: value for name, value in observation.items() if name not in IMAGE_FIELDS},
                           **{f"{name}_sha256": image_digest(observation[name]) for name in IMAGE_FIELDS}}
        values = {
            **observation, "executed_action": np.asarray(action, dtype=np.float32),
            "step_index": np.asarray(step_index, dtype=np.int64),
            "sim_timestamp": np.asarray(sim_timestamp, dtype=np.float64),
            "sim_state": np.asarray(sim_state), "phase": np.asarray(phase, dtype=np.uint8),
            "camera_pose": np.asarray(camera_pose),
        }
        if self.count and set(values) != set(self.steps):
            raise ValueError("Episode fields changed between steps")
        for name, value in values.items():
            value = np.asarray(value)
            if name not in self.steps:
                self.steps.create_dataset(
                    name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                    chunks=(1, *value.shape), dtype=value.dtype,
                    compression="lzf", shuffle=True,
                )
            dataset = self.steps[name]
            dataset.resize(self.count + 1, axis=0)
            dataset[self.count] = value
        self.count += 1
        self.file.flush()

    def finish(self, success, reason, final_sim_state, inference_seconds, error=None):
        self.file.attrs["success"] = bool(success)
        self.file.attrs["termination_reason"] = reason
        self.file.attrs["error"] = error or ""
        self.file.attrs["complete"] = True
        self.file.create_dataset("final_sim_state", data=final_sim_state)
        self.file.create_dataset("inference_seconds", data=inference_seconds)
        self.file.flush()
        self.file.close()
        with self.temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(self.temporary, self.destination)
        self.temporary.unlink()
        return self.destination


def episode_index(directory):
    result = []
    for path in sorted(Path(directory).glob("*.h5")):
        if path.name.endswith(".partial.h5"):
            continue
        with h5py.File(path, "r") as stream:
            if not stream.attrs.get("complete", False):
                continue
            metadata = json.loads(stream.attrs["metadata"])
            result.append({
                "path": str(path), "episode_id": str(stream.attrs["episode_id"]),
                "success": bool(stream.attrs["success"]),
                "termination_reason": str(stream.attrs["termination_reason"]),
                **metadata,
            })
    return result
