import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np

from depth_policy.common import atomic_json, load_config, sha256_file
from depth_policy.episodes import IMAGE_FIELDS, episode_index, image_digest


def validate_episode(path, manifest):
    with h5py.File(path, "r") as stream:
        assert stream.attrs["complete"], "Incomplete episode"
        metadata = json.loads(stream.attrs["metadata"])
        entry = manifest["states"][metadata["initial_state_id"]]
        assert metadata["split"] == entry["split"], "Split mismatch"
        assert metadata["initial_state_sha256"] == entry["sha256"], "Initial-state mismatch"
        assert metadata["depth_units"] == "m", "Depth is not metric"
        assert metadata["orientation"] == "flip_both_axes", "Unknown image orientation"
        steps = stream["steps"]
        count = len(steps["state"])
        assert count > 0, "Empty episode"
        assert all(len(dataset) == count for dataset in steps.values()), "Unequal field lengths"
        assert steps["state"].shape == (count, 8), "State shape mismatch"
        assert steps["executed_action"].shape == (count, 7), "Action shape mismatch"
        np.testing.assert_array_equal(steps["step_index"][:], np.arange(count))
        np.testing.assert_allclose(np.diff(steps["sim_timestamp"][:]),
                                   1.0 / metadata["control_freq"], atol=1e-8)
        phase = steps["phase"][:]
        np.testing.assert_array_equal(phase, np.arange(count) >= metadata["config"]["settle_steps"])
        for name in ["state", "executed_action", "sim_state", "camera_pose"]:
            assert np.isfinite(steps[name][:]).all(), f"Nonfinite {name}"
        depth_ranges = {}
        compact = not metadata.get("record_images", True)
        if compact:
            assert metadata.get("evaluation_only")
            first = stream["first_policy_observation"]
            first_index = int(first.attrs["step_index"])
            assert first_index == int(np.flatnonzero(phase == 1)[0])
            np.testing.assert_array_equal(first["state"][:], steps["state"][first_index])
            for name in IMAGE_FIELDS:
                hashes = steps[f"{name}_sha256"]
                assert hashes.shape == (count, 32) and hashes.dtype == np.uint8
                np.testing.assert_array_equal(image_digest(first[name][:]), hashes[first_index])
        for camera in ["agentview", "wrist"]:
            depth = first[f"depth_{camera}"][:][None] if compact else steps[f"depth_{camera}"]
            rgb = first[f"rgb_{camera}"][:][None] if compact else steps[f"rgb_{camera}"]
            resolution = metadata["config"]["resolution"]
            image_count = 1 if compact else count
            assert depth.shape == (image_count, resolution, resolution)
            assert depth.dtype == np.float32
            assert rgb.shape == (image_count, resolution, resolution, 3) and rgb.dtype == np.uint8
            minimum, maximum = float("inf"), 0.0
            for frame in depth:
                assert np.isfinite(frame).all() and (frame > 0).all()
                minimum, maximum = min(minimum, float(frame.min())), max(maximum, float(frame.max()))
            depth_ranges[camera] = [minimum, maximum]
        success = bool(stream.attrs["success"])
        assert success == (stream.attrs["termination_reason"] == "success")
        return {"path": str(path), "frames": count, "demonstration_frames": int(phase.sum()),
                "record_images": not compact,
                "success": success, "depth_range_m": depth_ranges,
                "initial_state_id": metadata["initial_state_id"], "split": metadata["split"],
                "sha256": sha256_file(path)}


def preview(path, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(path, "r") as stream:
        selected = np.flatnonzero(stream["steps/phase"][:] == 1)
        indices = [int(selected[0]), int(selected[len(selected) // 2]), int(selected[-1])] if len(selected) else [0]
        figure, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
        for row, index in enumerate(indices):
            for camera_index, camera in enumerate(["agentview", "wrist"]):
                axes[row, camera_index * 2].imshow(stream[f"steps/rgb_{camera}"][index])
                image = axes[row, camera_index * 2 + 1].imshow(
                    stream[f"steps/depth_{camera}"][index], vmin=0, vmax=3, cmap="viridis")
                figure.colorbar(image, ax=axes[row, camera_index * 2 + 1], label="depth (m)")
                axes[row, camera_index * 2].set_title(f"{camera} RGB, step {index}")
                axes[row, camera_index * 2 + 1].set_title(f"{camera} metric depth")
            for axis in axes[row]:
                axis.axis("off")
        figure.tight_layout()
        figure.savefig(destination, dpi=120)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--data", default="data/teacher")
    parser.add_argument("--report", default="reports/data-validation.json")
    parser.add_argument("--preview")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads(Path(config["manifest"]).read_text())
    episodes = episode_index(args.data)
    if not episodes:
        raise ValueError("No completed episodes to validate")
    records = [validate_episode(item["path"], manifest) for item in episodes]
    counts = collections.Counter(item["termination_reason"] for item in episodes)
    summary = {"episodes": len(records), "termination_counts": dict(counts),
               "frames": sum(item["frames"] for item in records),
               "unique_initial_states": len({item["initial_state_id"] for item in records}),
               "records": records}
    atomic_json(args.report, summary)
    if args.preview:
        preview(episodes[0]["path"], args.preview)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))


if __name__ == "__main__":
    main()
