import argparse
import json
from pathlib import Path
import signal

import numpy as np
import torch

from depth_policy.collect import run_episode
from depth_policy.common import atomic_json, load_config, sha256_file
from depth_policy.data import encode_language, preprocess_images
from depth_policy.episodes import episode_index
from depth_policy.manifest import load_manifest
from depth_policy.model import StudentPolicy
from depth_policy.simulation import make_env, task_and_states


class Student:
    def __init__(self, path, device):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.statistics = checkpoint["statistics"]
        self.vocabulary = checkpoint["vocabulary"]
        self.device = device
        self.model = StudentPolicy(self.config["modality"], len(self.vocabulary), self.config["horizon"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.metadata = {"policy_config": f"student_{self.config['modality']}",
                         "checkpoint_path": str(Path(path).resolve()), "sha256": sha256_file(path),
                         "training_seed": self.config["seed"], "training_step": checkpoint["step"]}

    @torch.no_grad()
    def infer(self, observation, instruction):
        modality = self.config["modality"]
        images = None
        if modality != "state":
            images = preprocess_images(
                observation[f"{modality}_agentview"][None], observation[f"{modality}_wrist"][None],
                modality, self.config["image_size"], self.config["depth_max"]
            ).half().to(self.device)
        state = (observation["state"] - np.asarray(self.statistics["state_mean"])) / np.asarray(
            self.statistics["state_std"])
        state = torch.tensor(state, dtype=torch.float32, device=self.device)[None]
        tokens = torch.tensor(encode_language(instruction, self.vocabulary), device=self.device)[None]
        prediction = self.model(state, tokens, images)[0].cpu().numpy()
        return (prediction * np.asarray(self.statistics["action_std"]) +
                np.asarray(self.statistics["action_mean"])).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = Student(args.checkpoint, args.device)
    config = load_config(args.config)
    config["teacher_replan_steps"] = args.replan_steps
    _, states, initial_file = task_and_states(config)
    manifest = load_manifest(config, states, initial_file)
    environment, task = make_env(config, config["seed"])
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    previous = episode_index(args.output)
    if any(item["teacher"] != policy.metadata or item["split"] != args.split for item in previous):
        raise ValueError("Evaluation output belongs to a different model or split")
    done = {(item["initial_state_id"], item["seed"]) for item in previous
            if item["termination_reason"] not in {"exception", "interrupted"}}
    try:
        for repeat in range(args.repeats):
            for entry in manifest["states"]:
                if entry["split"] != args.split or stop["requested"]:
                    continue
                seed = config["seed"] + entry["id"] + repeat * len(states)
                if (entry["id"], seed) in done:
                    continue
                metadata = {"task_name": config["task"], "suite": config["suite"],
                            "split": args.split, "initial_state_id": entry["id"],
                            "initial_state_sha256": entry["sha256"], "seed": seed,
                            "teacher": policy.metadata, "depth_units": "m", "config": config,
                            "control_freq": config["control_freq"], "orientation": "flip_both_axes",
                            "manifest_sha256": sha256_file(config["manifest"])}
                run_episode(environment, task, states[entry["id"]], metadata, config, policy, args.output, stop)
    finally:
        environment.close()
    records = episode_index(args.output)
    result = {"attempts": len(records), "successes": sum(item["success"] for item in records),
              "unique_initial_states": len({item["initial_state_id"] for item in records}),
              "policy": policy.metadata, "split": args.split}
    result["success_rate"] = result["successes"] / result["attempts"] if result["attempts"] else None
    atomic_json(Path(args.output) / "summary.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
