"""Evaluate the frozen BC, optionally with a deterministic residual PPO actor."""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="airbot_rl/eval/bc")
parser.add_argument("--checkpoint", help="BC checkpoint; defaults to CanEnvCfg.checkpoint")
parser.add_argument("--ppo", help="rsl_rl model_N.pt; use its deterministic actor mean as the residual")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.sim import SimulationContext  # noqa: E402

from airbot_rl.env import CanEnv, CanEnvCfg  # noqa: E402
from airbot_rl.scene import CAN_HEIGHT, CAN_RADIUS  # noqa: E402


def load_actor(path, env, obs_dict, device):
    from rsl_rl.models import MLPModel

    obs = TensorDict(obs_dict, batch_size=[env.num_envs])
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        env.cfg.action_space,
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    ).to(device)
    actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor_state_dict"])
    return actor.eval()


def main():
    cfg = CanEnvCfg()
    cfg.seed, cfg.scene.num_envs = args.seed, args.episodes
    if args.checkpoint:
        cfg.checkpoint = str(Path(args.checkpoint).resolve())
    env = CanEnv(cfg)
    obs_dict, _ = env.reset()
    actor = load_actor(args.ppo, env, obs_dict, env.device) if args.ppo else None
    out = Path(args.output)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    toppled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    excess = torch.zeros(env.num_envs, device=env.device)
    length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    approach_min = torch.full((env.num_envs,), float("inf"), device=env.device)
    approach_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    records, decision = [None] * env.num_envs, 0
    while active.any():
        running = active.clone()
        can_pos = env.can.data.root_pos_w - env.scene.env_origins
        cos = env.can_tilt_cos().abs()
        lowest = can_pos[:, 2] - CAN_HEIGHT / 2 * cos - CAN_RADIUS * (1 - cos.square()).clamp_min(0).sqrt()
        grasp = env.grasp_point()
        horizontal = (grasp[:, :2] - can_pos[:, :2]).square().sum(1).sqrt()
        near = ((horizontal >= 0.04) & (horizontal <= 0.07) & (lowest < 0.01) & running)
        approach_min = torch.where(near, torch.minimum(approach_min, grasp[:, 2]), approach_min)
        approach_seen |= near

        if actor:
            observations = TensorDict(obs_dict, batch_size=[env.num_envs])
            with torch.inference_mode():
                action = actor(observations)
        else:
            action = zero
        obs_dict, _, success, timeout, _ = env.step(action)
        touched |= env.touched & running
        toppled |= env.episode_toppled & running
        excess += env.excess * running
        length += running
        done = success | timeout
        for i in (done & running).nonzero().flatten().tolist():
            records[i] = {
                "success": bool(success[i]),
                "decisions": int(length[i]),
                "contact": bool(touched[i]),
                "toppled": bool(toppled[i]),
                "clip_excess": float(excess[i]),
                "approach_entered": bool(approach_seen[i]),
                "approach_height": float(approach_min[i]) if approach_seen[i] else None,
            }
            active[i] = False
        if decision % 10 == 0:
            depth = env.bc.last_depth[0, :, 0].cpu().numpy()
            Image.fromarray((np.concatenate(depth, 1) * 255).astype(np.uint8)).save(out / f"frames/{decision:04d}.png")
        decision += 1

    episodes = [record for record in records if record is not None]
    heights = [record["approach_height"] for record in episodes if record["approach_height"] is not None]
    summary = {key: sum(record[key] for record in episodes) / len(episodes)
               for key in ("success", "contact", "toppled")}
    summary.update(
        episodes=len(episodes),
        mean_clip_excess=sum(record["clip_excess"] for record in episodes) / len(episodes),
        approach_height_median=float(np.median(heights)) if heights else None,
        approach_no_entry_rate=1 - len(heights) / len(episodes),
    )
    payload = {
        "checkpoint": str(env.bc.policy.checkpoint_path),
        "checkpoint_sha256": env.bc.policy.checkpoint_sha256,
        "seed": args.seed,
        "summary": summary,
        "episodes": episodes,
    }
    if args.ppo:
        payload["ppo"] = str(Path(args.ppo).resolve())
    (out / "summary.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary), flush=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
