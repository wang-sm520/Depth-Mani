"""Exercise trajectory boundaries and timestamp alignment without robot hardware."""

from fractions import Fraction
import hashlib
import json

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from airbot_depth.data import AirbotDataset, fit_statistics, load_manifest, select_episodes
from airbot_depth.lerobot import LeRobotSource


HEAD = "observation.images.head"
WRIST = "observation.images.wrist"
PROMPT = "pick up the bag"
NATIVE_NAMES = [f"joint{i}.pos" for i in range(1, 7)] + ["eef.pos"]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_video(path, intensities, pts=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        stream.codec_context.thread_count = 1
        for index, intensity in enumerate(intensities):
            pixels = np.full((16, 16, 3), intensity, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index if pts is None else pts[index]
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def lerobot_fixture(tmp_path):
    """Different camera offsets and a file boundary distinguish wrong RGB mappings."""
    root = tmp_path / "source"
    features = {
        key: {"dtype": "float32", "shape": [7], "names": list(NATIVE_NAMES)}
        for key in ("observation.state", "action")
    }
    features.update({key: {"dtype": "video", "shape": [16, 16, 3],
                           "info": {"video.is_depth_map": False}}
                     for key in (HEAD, WRIST)})
    info = {"codebase_version": "v3.0", "robot_type": "airbot_play_follower",
            "total_episodes": 2, "total_frames": 5, "fps": 10, "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    _write_json(root / "meta/info.json", info)
    pq.write_table(pa.table({"task_index": [0], "__index_level_0__": [PROMPT]}),
                   root / "meta/tasks.parquet")
    states = np.arange(35, dtype=np.float32).reshape(5, 7) / 100
    actions = states + np.float32(0.003)
    table = pa.table({
        "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 7)),
        "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 7)),
        "episode_index": [0, 0, 0, 1, 1], "frame_index": [0, 1, 2, 0, 1],
        "index": list(range(5)), "task_index": [0] * 5,
        "timestamp": pa.array([0, 0.1, 0.2, 0, 0.1], type=pa.float32()),
    })
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(table, data_path)
    episodes = []
    for index, start, length in [(0, 0, 3), (1, 3, 2)]:
        row = {"episode_index": index, "length": length,
               "dataset_from_index": start, "dataset_to_index": start + length,
               "data/chunk_index": 0, "data/file_index": 0, "tasks": [PROMPT]}
        for key, video_start, video_file in [(HEAD, [0.2, 0.5][index], 0),
                                            (WRIST, [0.0, 0.1][index], index)]:
            prefix = f"videos/{key}"
            row.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": video_file,
                        f"{prefix}/from_timestamp": video_start,
                        f"{prefix}/to_timestamp": video_start + length / 10})
        episodes.append(row)
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), metadata_path)
    _write_video(root / f"videos/{HEAD}/chunk-000/file-000.mp4", [20, 30, 40, 50, 60, 70, 80])
    _write_video(root / f"videos/{WRIST}/chunk-000/file-000.mp4", [90, 100, 110])
    _write_video(root / f"videos/{WRIST}/chunk-000/file-001.mp4", [120, 130, 140])
    return {"root": root, "info": info, "episodes": episodes, "table": table,
            "metadata_path": metadata_path, "data_path": data_path,
            "states": states, "actions": actions}


def test_lerobot_preserves_native_targets_and_seeks_each_camera_offset(lerobot_fixture):
    fixture = lerobot_fixture
    source = LeRobotSource(fixture["root"])
    episode = source.load_episode(1)
    np.testing.assert_array_equal(episode["state"], fixture["states"][3:])
    np.testing.assert_array_equal(episode["action"], fixture["actions"][3:])
    # Action must remain the recorded absolute target, without a second delta transform.
    assert not np.array_equal(episode["action"], episode["action"] - episode["state"])
    for key, expected_pts, expected_pixels in [(HEAD, [0.5, 0.6], [70, 80]),
                                               (WRIST, [0.1, 0.2], [130, 140])]:
        decoded = list(source.rgb_frames(1, key, episode["timestamp"]))
        np.testing.assert_allclose([stamp for _, stamp in decoded], expected_pts, atol=1e-6)
        np.testing.assert_allclose([rgb.mean() for rgb, _ in decoded], expected_pixels, atol=2)


def test_lerobot_rejects_video_that_ends_before_episode(lerobot_fixture):
    fixture = lerobot_fixture
    path = fixture["root"] / f"videos/{HEAD}/chunk-000/file-000.mp4"
    _write_video(path, [20, 30, 40, 50, 60, 70])
    source = LeRobotSource(fixture["root"])
    with pytest.raises(ValueError, match="ended early"):
        list(source.rgb_frames(1, HEAD, source.load_episode(1)["timestamp"]))


def test_lerobot_rejects_rgb_timestamp_gap(lerobot_fixture):
    fixture = lerobot_fixture
    path = fixture["root"] / f"videos/{HEAD}/chunk-000/file-000.mp4"
    _write_video(path, [20, 30, 40, 50, 60, 70, 80], pts=[0, 1, 2, 3, 4, 6, 7])
    source = LeRobotSource(fixture["root"])
    with pytest.raises(ValueError, match="Missing RGB frame|FPS differs"):
        list(source.rgb_frames(1, HEAD, source.load_episode(1)["timestamp"]))


def test_lerobot_rejects_video_request_from_another_episode(lerobot_fixture):
    source = LeRobotSource(lerobot_fixture["root"])
    # Episode zero starts at video 0.2 and ends at 0.5. The requested 0.3 would
    # silently select episode one's first RGB if episode bounds were unchecked.
    with pytest.raises(ValueError, match="episode|interval|range|bound"):
        list(source.rgb_frames(0, HEAD, np.asarray([0.3])))


def test_lerobot_rejects_overlapping_episode_video_mapping(lerobot_fixture):
    fixture = lerobot_fixture
    row = fixture["episodes"][1]
    row[f"videos/{HEAD}/from_timestamp"] = 0.3
    row[f"videos/{HEAD}/to_timestamp"] = 0.5
    pq.write_table(pa.Table.from_pylist(fixture["episodes"]), fixture["metadata_path"])
    with pytest.raises(ValueError, match="overlap|interval|mapping"):
        LeRobotSource(fixture["root"])


@pytest.mark.parametrize("field", ["observation.state", "action"])
def test_lerobot_rejects_reordered_joint_names(lerobot_fixture, field):
    fixture = lerobot_fixture
    names = fixture["info"]["features"][field]["names"]
    names[0], names[1] = names[1], names[0]
    _write_json(fixture["root"] / "meta/info.json", fixture["info"])
    with pytest.raises(ValueError, match="names|order|semantics"):
        LeRobotSource(fixture["root"])


def test_lerobot_rejects_wrong_existing_parquet_mapping(lerobot_fixture):
    fixture = lerobot_fixture
    wrong_path = fixture["data_path"].with_name("file-001.parquet")
    pq.write_table(fixture["table"].slice(0, 3), wrong_path)
    fixture["episodes"][1]["data/file_index"] = 1
    pq.write_table(pa.Table.from_pylist(fixture["episodes"]), fixture["metadata_path"])
    source = LeRobotSource(fixture["root"])
    with pytest.raises(ValueError, match="Parquet length mismatch"):
        source.load_episode(1)


def test_lerobot_rejects_shifted_global_row_indices(lerobot_fixture):
    fixture = lerobot_fixture
    table = fixture["table"]
    table = table.set_column(table.schema.get_field_index("index"), "index", pa.array([1, 2, 3, 4, 5]))
    pq.write_table(table, fixture["data_path"])
    source = LeRobotSource(fixture["root"])
    with pytest.raises(ValueError, match="Global row indices"):
        source.load_episode(0)


@pytest.fixture
def converted_fixture(tmp_path):
    root = tmp_path / "converted"
    root.mkdir()
    episodes, raw = [], []
    for index, (length, split, base) in enumerate([(3, "train", 0), (2, "validation", 100)]):
        states = base + np.arange(length * 7, dtype=np.float32).reshape(length, 7) / 10
        actions = states + np.float32(0.25)
        depth = np.full((length, 2, 2, 8, 8), 0.2, dtype=np.float32)
        depth[:, :, 1] = 1
        path = root / f"episode-{index}.h5"
        with h5py.File(path, "w") as stream:
            stream.attrs["episode_index"] = index
            stream.attrs["instruction"] = PROMPT
            stream.create_dataset("state", data=states)
            stream.create_dataset("action", data=actions)
            stream.create_dataset("depth", data=depth)
        episodes.append({"episode_index": index, "path": path.name, "split": split,
                         "frames": length, "instruction": PROMPT,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        raw.append({"state": states, "action": actions})
    manifest = {"schema_version": 1, "status": "complete", "prompt": PROMPT,
                "action_semantics": "absolute_joint_position", "episodes": episodes,
                "source": {"selected_episode_indices": [0, 1]}, "camera_keys": [HEAD, WRIST],
                "state_dim": 7, "action_dim": 7, "depth_config": {"image_size": 8}}
    _write_json(root / "manifest.json", manifest)
    return {"root": root, "manifest": manifest, "raw": raw}


def test_action_chunks_stop_at_episode_end(converted_fixture):
    fixture = converted_fixture
    manifest = load_manifest(fixture["root"])
    statistics = {f"{kind}_{stat}": [value] * 7
                  for kind in ("state", "action") for stat, value in [("mean", 0), ("std", 1)]}
    vocabulary = {"<pad>": 0, "<unk>": 1, "pick": 2, "up": 3, "the": 4, "bag": 5}
    dataset = AirbotDataset(fixture["root"], manifest["episodes"], statistics, vocabulary, horizon=4)
    tail = dataset[2]
    np.testing.assert_array_equal(tail["action"][0], fixture["raw"][0]["action"][-1])
    assert tail["mask"].tolist() == [True, False, False, False]
    assert torch.count_nonzero(tail["action"][1:]) == 0
    next_episode = dataset[3]
    np.testing.assert_array_equal(next_episode["action"][:2], fixture["raw"][1]["action"])
    assert next_episode["mask"].tolist() == [True, True, False, False]
    assert dataset[4]["mask"].tolist() == [True, False, False, False]
    with pytest.raises(IndexError):
        dataset[5]


def test_statistics_include_training_tail_and_exclude_validation(converted_fixture):
    fixture = converted_fixture
    manifest = load_manifest(fixture["root"])
    statistics = fit_statistics(fixture["root"], select_episodes(manifest, "train"))
    for kind in ("state", "action"):
        np.testing.assert_allclose(statistics[f"{kind}_mean"], fixture["raw"][0][kind].mean(0), atol=1e-6)
        np.testing.assert_allclose(statistics[f"{kind}_std"], fixture["raw"][0][kind].std(0), atol=1e-6)
    with pytest.raises(ValueError, match="training episodes only"):
        fit_statistics(fixture["root"], manifest["episodes"])
    with pytest.raises(ValueError, match="training episodes only"):
        fit_statistics(fixture["root"], select_episodes(manifest, "validation"))


def test_manifest_rejects_same_episode_in_both_splits(converted_fixture):
    fixture = converted_fixture
    manifest = fixture["manifest"]
    manifest["episodes"][1]["episode_index"] = 0
    _write_json(fixture["root"] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Duplicate|incomplete"):
        load_manifest(fixture["root"])


def test_manifest_requires_held_out_episodes(converted_fixture):
    fixture = converted_fixture
    fixture["manifest"]["episodes"][1]["split"] = "train"
    _write_json(fixture["root"] / "manifest.json", fixture["manifest"])
    with pytest.raises(ValueError, match="nonempty.*train.*validation"):
        load_manifest(fixture["root"])


def test_manifest_rejects_training_prompt_that_differs_from_deployment(converted_fixture):
    fixture = converted_fixture
    fixture["manifest"]["episodes"][0]["instruction"] = "put down the bag"
    _write_json(fixture["root"] / "manifest.json", fixture["manifest"])
    with pytest.raises(ValueError, match="prompt|instruction"):
        load_manifest(fixture["root"])
