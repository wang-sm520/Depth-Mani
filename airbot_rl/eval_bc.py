"""Zero-residual rollout: the frozen BC alone in simulation, the go/no-go check before PPO."""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="airbot_rl/eval/bc")
parser.add_argument("--checkpoint", help="BC checkpoint to evaluate; defaults to CanEnvCfg.checkpoint")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.sim import SimulationContext  # noqa: E402

from airbot_rl.env import CanEnv, CanEnvCfg  # noqa: E402


def main():
    cfg = CanEnvCfg()
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    if args.checkpoint:
        cfg.checkpoint = str(Path(args.checkpoint).resolve())
    env = CanEnv(cfg)
    env.reset()
    out = Path(args.output)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    excess = torch.zeros(env.num_envs, device=env.device)
    length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episodes, decision = [], 0
    while len(episodes) < args.episodes:
        toppled = env.toppled.clone()
        _, _, success, timeout, _ = env.step(zero)
        touched |= env.touched
        excess += env.excess
        length += 1
        if decision % 10 == 0:  # env 0 film strip: the BC's head | wrist depth input
            depth = env.bc.last_depth[0, :, 0].cpu().numpy()
            Image.fromarray((np.concatenate(depth, 1) * 255).astype(np.uint8)).save(out / f"frames/{decision:04d}.png")
        for i in (success | timeout).nonzero().flatten().tolist():
            episodes.append({"success": bool(success[i]), "decisions": int(length[i]),
                             "contact": bool(touched[i]), "toppled": bool(toppled[i]), "clip_excess": float(excess[i])})
            touched[i], excess[i], length[i] = False, 0.0, 0
        decision += 1
    episodes = episodes[:args.episodes]
    summary = {key: sum(e[key] for e in episodes) / len(episodes) for key in ("success", "contact", "toppled")}
    summary.update(episodes=len(episodes), mean_clip_excess=sum(e["clip_excess"] for e in episodes) / len(episodes))
    (out / "summary.json").write_text(json.dumps({
        "checkpoint": str(env.bc.policy.checkpoint_path), "checkpoint_sha256": env.bc.policy.checkpoint_sha256,
        "seed": args.seed, "summary": summary, "episodes": episodes}, indent=1))
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
