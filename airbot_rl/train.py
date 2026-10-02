"""Residual PPO (rsl_rl) on the frozen BC. The actor's mean layer starts at zero, so iteration 0 is the BC exactly."""

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", help="frozen BC; defaults to CanEnvCfg.checkpoint")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", default="airbot_rl/runs/ppo")
parser.add_argument("--resume", help="rsl_rl checkpoint (model_N.pt) to continue from")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

from datetime import datetime  # noqa: E402

import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from airbot_rl.env import CanEnv, CanEnvCfg  # noqa: E402

TRAIN_CFG = {
    "num_steps_per_env": 32, "save_interval": 25, "experiment_name": "airbot_can_residual", "run_name": "",
    "logger": "tensorboard", "clip_actions": None, "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "actor": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "elu", "obs_normalization": True,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "elu", "obs_normalization": True},
    "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3e-4,
                  "schedule": "adaptive", "desired_kl": 0.01, "gamma": 0.99, "lam": 0.95, "entropy_coef": 0.0,
                  "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "max_grad_norm": 1.0,
                  "rnd_cfg": None, "symmetry_cfg": None},
}


def main():
    cfg = CanEnvCfg()
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    env = RslRlVecEnvWrapper(CanEnv(cfg))
    runner = OnPolicyRunner(env, {**TRAIN_CFG, "seed": args.seed, "max_iterations": args.iterations},
                            log_dir=f"{args.log_dir}/{datetime.now():%Y%m%d-%H%M%S}", device=env.device)
    if args.resume:
        runner.load(args.resume)  # restores actor, critic, normalisers, optimizer and the iteration counter
    else:
        mean = [m for m in runner.alg.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(mean.weight)
        torch.nn.init.zeros_(mean.bias)
    runner.learn(args.iterations, init_at_random_ep_len=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
