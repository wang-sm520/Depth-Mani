import os
from pathlib import Path
import sys

import numpy as np

from depth_policy.common import ROOT, atomic_json, orient_image, sha256_file, state_vector


CAMERAS = ("agentview", "robot0_eye_in_hand")


def configure_libero(config):
    root = Path(config["libero_root"]).resolve()
    sys.path.insert(0, str(root))
    directory = ROOT / "runtime/libero"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    benchmark_root = root / "libero/libero"
    paths = {"benchmark_root": str(benchmark_root), "datasets": str(ROOT / "data")}
    paths.update({key: str(benchmark_root / value) for key, value in {
        "bddl_files": "bddl_files", "init_states": "init_files", "assets": "assets"
    }.items()})
    atomic_json(directory / "config.yaml", paths)


def task_and_states(config):
    configure_libero(config)
    import torch
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    task = next(task for task in suite.tasks if task.name == config["task"])
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = np.asarray(torch.load(path, map_location="cpu", weights_only=False))
    return task, states, path


def make_env(config, seed):
    configure_libero(config)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task, _, _ = task_and_states(config)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if "evaluation_bddl" in config:
        if not config.get("evaluation_only"):
            raise ValueError("Custom BDDL requires explicit evaluation-only configuration")
        bddl_path = ROOT / config["evaluation_bddl"]
        if sha256_file(bddl_path) != config["evaluation_bddl_sha256"]:
            raise ValueError("Evaluation BDDL hash mismatch")
    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=config["resolution"], camera_widths=config["resolution"],
        camera_names=list(CAMERAS), camera_depths=True,
        control_freq=config["control_freq"], controller="OSC_POSE",
    )
    environment.seed(seed)
    if environment.env.control_freq != config["control_freq"]:
        raise ValueError("Control frequency mismatch")
    if environment.env.action_dim != 7:
        raise ValueError("Expected OSC_POSE 7-dimensional action")
    if "official_instruction" in config:
        instruction = config["official_instruction"]
        if environment.language_instruction.casefold() != instruction.casefold():
            environment.close()
            raise ValueError("Configured instruction differs from the loaded BDDL language")
        task = task._replace(language=instruction)
    return environment, task


def observe(environment, observation):
    from robosuite.utils.camera_utils import get_real_depth_map

    result = {"state": state_vector(observation)}
    for camera, name in zip(CAMERAS, ("agentview", "wrist"), strict=True):
        depth = get_real_depth_map(environment.sim, observation[f"{camera}_depth"])
        depth = orient_image(np.asarray(depth).squeeze(-1)).astype(np.float32)
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Invalid metric depth from {camera}")
        result[f"depth_{name}"] = depth
        result[f"rgb_{name}"] = orient_image(observation[f"{camera}_image"])
    return result


def camera_metadata(environment):
    model = environment.sim.model
    result = {}
    for name in CAMERAS:
        camera_id = model.camera_name2id(name)
        result[name] = {
            "fovy_degrees": float(model.cam_fovy[camera_id]),
            "near_m": float(model.vis.map.znear * model.stat.extent),
            "far_m": float(model.vis.map.zfar * model.stat.extent),
            "position_world": environment.sim.data.cam_xpos[camera_id].tolist(),
            "rotation_world": environment.sim.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
        }
    return result
