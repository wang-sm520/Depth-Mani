"""Read native absolute joint targets and timestamp-aligned LeRobot v3 RGB.

This importer deliberately does not import a robot SDK or change source files.
"""

import json
import math
from pathlib import Path

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from airbot_depth.common import sha256_file


DEFAULT_CAMERAS = ["observation.images.head", "observation.images.wrist"]


class LeRobotSource:
    def __init__(self, root, camera_keys=None):
        self.root = Path(root).resolve(strict=True)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("This importer requires LeRobot v3.0 episode/video metadata")
        if self.info["robot_type"] not in {"airbot_play_follower", "airbot_p6"}:
            raise ValueError("Unverified robot type; define its native action semantics first")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid dataset FPS")
        self.camera_keys = list(camera_keys or DEFAULT_CAMERAS)
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Camera keys must be nonempty and unique")
        features = self.info["features"]
        for field in ("observation.state", "action"):
            if features[field]["shape"] != [7] or features[field]["dtype"] != "float32":
                raise ValueError("Expected native six joints plus one gripper in float32")
        play_names = [f"joint{i}.pos" for i in range(1, 7)] + ["eef.pos"]
        p6_state = [f"joint_{i}.position_rad" for i in range(1, 7)] + ["gripper.position_m"]
        p6_action = [f"joint_{i}.command_rad" for i in range(1, 7)] + ["gripper.command_m"]
        names = (features["observation.state"].get("names"), features["action"].get("names"))
        if names not in ((play_names, play_names), (p6_state, p6_action)):
            raise ValueError("Unverified state/action joint names or ordering")
        for key in self.camera_keys:
            if features[key]["dtype"] != "video":
                raise ValueError(f"Expected video camera: {key}")
            if features[key].get("info", {}).get("video.is_depth_map", False):
                raise ValueError("Depth Anything requires RGB source videos")
        self.metadata_paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not self.metadata_paths:
            raise ValueError("Missing LeRobot v3 episode table")
        episodes = []
        for path in self.metadata_paths:
            table = pq.read_table(path)
            columns = [name for name in table.column_names if not name.startswith("stats/")]
            episodes.extend(table.select(columns).to_pylist())
        episodes.sort(key=lambda row: row["episode_index"])
        if len(episodes) != self.info["total_episodes"]:
            raise ValueError("Episode count differs from info.json")
        if len({row["episode_index"] for row in episodes}) != len(episodes):
            raise ValueError("Duplicate episode identity")
        self.episodes = {int(row["episode_index"]): row for row in episodes}
        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pylist()
        self.tasks = {}
        for task in tasks:
            text = task.get("task", task.get("__index_level_0__"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Unrecognized task text column")
            self.tasks[int(task["task_index"])] = text
        if len(self.tasks) != 1:
            raise ValueError("First AIRBOT adapter requires a single, explicit task prompt")
        self.prompt = next(iter(self.tasks.values()))
        expected_start = 0
        for row in episodes:
            length = int(row["length"])
            if length < 1 or row["dataset_from_index"] != expected_start:
                raise ValueError("Empty episode or gap/overlap in dataset indices")
            expected_start += length
            if row["dataset_to_index"] != expected_start or row["tasks"] != [self.prompt]:
                raise ValueError("Episode boundary or instruction mismatch")
            for key in self.camera_keys:
                prefix = f"videos/{key}"
                start = float(row[f"{prefix}/from_timestamp"])
                end = float(row[f"{prefix}/to_timestamp"])
                if not np.isfinite([start, end]).all() or start < 0:
                    raise ValueError("Invalid video timestamp interval")
                if abs((end - start) - length / self.fps) > 1e-4:
                    raise ValueError("Video interval differs from episode length/FPS")
                if not self.video_path(row, key).is_file():
                    raise FileNotFoundError(self.video_path(row, key))
            if not self.data_path(row).is_file():
                raise FileNotFoundError(self.data_path(row))
        if expected_start != self.info["total_frames"]:
            raise ValueError("Frame count differs from info.json")
        intervals = {}
        for row in episodes:
            for key in self.camera_keys:
                intervals.setdefault((key, self.video_path(row, key)), []).append(
                    (row[f"videos/{key}/from_timestamp"], row[f"videos/{key}/to_timestamp"])
                )
        for spans in intervals.values():
            spans.sort()
            if any(first[1] > second[0] + 1e-4 for first, second in zip(spans, spans[1:])):
                raise ValueError("Overlapping episode video intervals would duplicate RGB labels")
        self._table_path = None
        self._table = None

    def data_path(self, episode):
        return self.root / self.info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )

    def video_path(self, episode, key):
        return self.root / self.info["video_path"].format(
            video_key=key, chunk_index=episode[f"videos/{key}/chunk_index"],
            file_index=episode[f"videos/{key}/file_index"],
        )

    def load_episode(self, episode_index):
        row = self.episodes[episode_index]
        path = self.data_path(row)
        if self._table_path != path:
            self._table = pq.read_table(path)
            self._table_path = path
        table = self._table.filter(pc.equal(self._table["episode_index"], episode_index))
        length = int(row["length"])
        if len(table) != length:
            raise ValueError(f"Episode {episode_index}: Parquet length mismatch")
        values = {}
        for output, field in [("state", "observation.state"), ("action", "action")]:
            array = np.asarray(table[field].to_pylist(), dtype=np.float32)
            if array.shape != (length, 7) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {field} in episode {episode_index}")
            values[output] = array
        values["timestamp"] = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        values["frame_index"] = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        values["source_index"] = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        if not np.array_equal(values["frame_index"], np.arange(length)):
            raise ValueError("Frames must be contiguous and episode-local")
        if not np.array_equal(values["source_index"],
                              np.arange(row["dataset_from_index"], row["dataset_to_index"])):
            raise ValueError("Global row indices do not match episode metadata")
        if not np.allclose(values["timestamp"], np.arange(length) / self.fps, atol=1e-4, rtol=0):
            raise ValueError("Nonuniform or mismatched observation/action timestamps")
        if set(table["task_index"].to_pylist()) != set(self.tasks):
            raise ValueError("Frame task labels differ from the fixed prompt")
        return values

    def rgb_frames(self, episode_index, key, timestamps):
        """Yield RGB plus actual video PTS; reject missing or misaligned frames."""
        row = self.episodes[episode_index]
        relative = np.asarray(timestamps, dtype=np.float64)
        if relative.ndim != 1 or not len(relative):
            raise ValueError("Expected an episode-local timestamp vector")
        if not np.isfinite(relative).all() or np.any(np.diff(relative) <= 0):
            raise ValueError("Expected strictly increasing finite video timestamps")
        if relative[0] < -1e-4 or relative[-1] > (row["length"] - 1) / self.fps + 1e-4:
            raise ValueError("RGB timestamps are outside the episode boundary")
        if not np.allclose(relative, np.rint(relative * self.fps) / self.fps, atol=1e-4, rtol=0):
            raise ValueError("RGB timestamps must match dataset frame times")
        if key not in self.camera_keys:
            raise ValueError("Camera not declared in source contract")
        targets = float(row[f"videos/{key}/from_timestamp"]) + relative
        tolerance = min(0.002, 0.1 / self.fps)
        with av.open(str(self.video_path(row, key))) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            if abs(float(stream.average_rate) - self.fps) > 1e-6:
                raise ValueError("Video FPS differs from dataset FPS")
            container.seek(math.floor(float(targets[0]) / float(stream.time_base)),
                           stream=stream, backward=True, any_frame=False)
            index = 0
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Video frame has no presentation timestamp")
                timestamp = float(frame.pts * frame.time_base)
                target = float(targets[index])
                if timestamp < target - tolerance:
                    continue
                if abs(timestamp - target) > tolerance:
                    raise ValueError(f"Missing RGB frame: episode={episode_index}, camera={key}, "
                                     f"wanted={target:.6f}, decoded={timestamp:.6f}")
                rgb = frame.to_ndarray(format="rgb24")
                expected_shape = self.info["features"][key]["shape"]
                if list(rgb.shape) != expected_shape or rgb.dtype != np.uint8:
                    raise ValueError("Decoded video geometry differs from its metadata")
                yield rgb, timestamp
                index += 1
                if index == len(targets):
                    return
            raise ValueError(f"Video ended early: episode={episode_index}, camera={key}")

    def fingerprint(self, episode_indices):
        paths = {self.root / "meta/info.json", self.root / "meta/tasks.parquet", *self.metadata_paths}
        for index in episode_indices:
            row = self.episodes[index]
            paths.add(self.data_path(row))
            paths.update(self.video_path(row, key) for key in self.camera_keys)
        return {"root": str(self.root), "robot_type": self.info["robot_type"],
                "total_episodes": self.info["total_episodes"],
                "total_frames": self.info["total_frames"],
                "selected_episode_indices": list(episode_indices),
                "files": [{"path": str(path.relative_to(self.root)),
                           "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                          for path in sorted(paths)]}
