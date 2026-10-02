"""Record BC rollouts in sim: one mp4 per env (side view, plus the head and wrist images the BC sees) and
rollout.npz with every decision's state, BC chunk, grasp point, can position and BC depth input.

One video frame per policy decision (4 targets). Residual is zero, so this is the frozen BC alone.
"""

import argparse
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--decisions", type=int, default=250)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="airbot_rl/eval/video")
parser.add_argument("--checkpoint", help="BC checkpoint to record; defaults to CanEnvCfg.checkpoint")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import av  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

from airbot_rl.env import CanEnv, CanEnvCfg  # noqa: E402

PITCH, YAW = math.radians(12), math.radians(90)  # side view: looking +y, tilted down
SIDE_ROT = (math.cos(YAW / 2) * math.cos(PITCH / 2), -math.sin(YAW / 2) * math.sin(PITCH / 2),
            math.cos(YAW / 2) * math.sin(PITCH / 2), math.sin(YAW / 2) * math.cos(PITCH / 2))


def label(image, text):
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main():
    cfg = CanEnvCfg()
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    if args.checkpoint:
        cfg.checkpoint = str(Path(args.checkpoint).resolve())
    cfg.scene.side_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/side_cam", data_types=["rgb"], width=960, height=540,
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.35, -1.4, 0.55), rot=SIDE_ROT, convention="world"))
    env = CanEnv(cfg)
    env.reset()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    writers = []
    for i in range(env.num_envs):
        container = av.open(str(out / f"bc_rollout_env{i}.mp4"), "w")
        stream = container.add_stream("h264", rate=6)
        stream.width, stream.height, stream.pix_fmt = 960, 812, "yuv420p"
        writers.append((container, stream))
    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    log = {key: [] for key in ("state", "chunk", "grasp", "can", "depth", "done")}
    for decision in range(args.decisions):
        log["state"].append(env.native_state().cpu().numpy())
        log["chunk"].append(env.bc_chunk.cpu().numpy())
        log["grasp"].append(env.grasp_point().cpu().numpy())
        log["can"].append((env.can.data.root_pos_w - env.scene.env_origins).cpu().numpy())
        log["depth"].append(env.bc.last_depth.cpu().numpy().astype(np.float16))
        log["done"].append(done.cpu().numpy())
        # BC inputs, cropped from the 128 x 128 letterbox to their 128 x 72 content (both views are 16:9).
        depth = (env.bc.last_depth[:, :, 0, 28:100].cpu().numpy() * 255).astype(np.uint8)
        side = env.scene["side_cam"].data.output["rgb"].cpu().numpy()
        state = env.native_state().cpu().numpy()
        for i, (container, stream) in enumerate(writers):
            if done[i]:
                continue
            bottom = np.repeat(np.concatenate([cv2.resize(view, (480, 270), interpolation=cv2.INTER_NEAREST)
                                               for view in depth[i]], 1)[..., None], 3, 2)
            frame = np.concatenate([label(side[i].copy(), f"decision {decision}  j2 {state[i, 1]:+.2f} rad  "
                                                         f"gripper {state[i, 6] * 100:.1f} cm"), bottom,
                                    np.zeros((2, 960, 3), np.uint8)], 0)
            label(frame[540:], "head DA2 (BC input)")
            label(frame[540:, 480:], "wrist depth (BC input)")
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        _, _, success, timeout, _ = env.step(zero)
        for i in (success | timeout).nonzero().flatten().tolist():
            if not done[i]:
                print(f"env {i}: {'success' if success[i] else 'timeout'} after {decision + 1} decisions", flush=True)
        done |= success | timeout
        if done.all():
            break
    np.savez_compressed(out / "rollout.npz", **{key: np.stack(value) for key, value in log.items()})
    for container, stream in writers:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    sim_utils.SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
