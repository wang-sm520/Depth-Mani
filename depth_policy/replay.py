import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from depth_policy.common import atomic_json
from depth_policy.episodes import IMAGE_FIELDS, image_digest
from depth_policy.simulation import make_env, observe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    parser.add_argument("--report", default="reports/replay.json")
    parser.add_argument("--video")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("Playback speed must be positive")
    if args.video and Path(args.video).exists():
        raise FileExistsError("Video destination already exists")
    with h5py.File(args.episode, "r") as stream:
        metadata = json.loads(stream.attrs["metadata"])
        environment, _ = make_env(metadata["config"], metadata["seed"])
        errors = {"sim_state": 0.0, "robot_state": 0.0, "depth_agentview": 0.0,
                  "depth_wrist": 0.0, "rgb_agentview": 0.0, "rgb_wrist": 0.0}
        compact = not metadata.get("record_images", True)
        if compact:
            errors = {"sim_state": 0.0, "robot_state": 0.0}
        writer = None
        try:
            if args.video:
                import imageio.v2 as imageio
                from depth_policy.replay_video import replay_frame

                Path(args.video).parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(args.video, fps=metadata["control_freq"] * args.speed,
                                            codec="libx264", quality=8, macro_block_size=16,
                                            ffmpeg_params=["-movflags", "+faststart"])
            environment.reset()
            if environment.sim.model.get_xml() != stream["model_xml"].asstr()[()]:
                raise ValueError("Reconstructed model XML differs; check environment version and seed")
            raw = environment.set_init_state(stream["initial_sim_state"][:])
            steps = stream["steps"]
            for index, action in enumerate(steps["executed_action"]):
                observation = observe(environment, raw)
                current = environment.get_sim_state()
                errors["sim_state"] = max(errors["sim_state"], float(np.max(np.abs(current - steps["sim_state"][index]))))
                errors["robot_state"] = max(errors["robot_state"], float(np.max(np.abs(observation["state"] - steps["state"][index]))))
                for name in ["depth_agentview", "depth_wrist"]:
                    if not compact:
                        errors[name] = max(errors[name], float(np.max(np.abs(observation[name] - steps[name][index]))))
                np.testing.assert_allclose(current, steps["sim_state"][index], atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(observation["state"], steps["state"][index], atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(environment.sim.data.time, steps["sim_timestamp"][index], atol=1e-8)
                for name in ["depth_agentview", "depth_wrist"]:
                    if not compact:
                        np.testing.assert_allclose(observation[name], steps[name][index], atol=1e-4, rtol=1e-4)
                if not compact:
                    for name in ("rgb_agentview", "rgb_wrist"):
                        actual = observation[name]
                        recorded = steps[name][index]
                        errors[name] = max(errors[name], float(np.max(np.abs(
                            actual.astype(np.int16) - recorded.astype(np.int16)))))
                        np.testing.assert_array_equal(actual, recorded)
                if compact:
                    for name in IMAGE_FIELDS:
                        np.testing.assert_array_equal(image_digest(observation[name]), steps[f"{name}_sha256"][index])
                if writer is not None:
                    writer.append_data(replay_frame(observation, index, float(steps["sim_timestamp"][index]),
                                                    metadata["instruction"],
                                                    "settling" if steps["phase"][index] == 0 else "recorded action replay"))
                raw, _, _, _ = environment.step(action.tolist())
            np.testing.assert_allclose(environment.get_sim_state(), stream["final_sim_state"][:], atol=1e-6, rtol=1e-6)
            assert bool(environment.check_success()) == bool(stream.attrs["success"])
            if writer is not None:
                final_frame = replay_frame(observe(environment, raw), len(steps["state"]),
                                           float(environment.sim.data.time), metadata["instruction"], "terminal",
                                           bool(environment.check_success()))
                for repeat in range(round(metadata["control_freq"] * args.speed)):
                    writer.append_data(final_frame)
                writer.close()
                writer = None
                imageio.imwrite(str(Path(args.video).with_suffix(".png")), final_frame)
            result = {"episode": args.episode, "frames": len(steps["state"]),
                      "passed": True, "maximum_absolute_errors": errors,
                      "video": args.video, "playback_speed": args.speed,
                      "image_hashes_verified": compact,
                      "full_rgbd_verified": not compact,
                      "success": bool(environment.check_success())}
            atomic_json(args.report, result)
            print(json.dumps(result))
        finally:
            if writer is not None:
                writer.close()
            environment.close()


if __name__ == "__main__":
    main()
