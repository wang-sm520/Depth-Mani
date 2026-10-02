import argparse
import collections
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import shutil

import h5py
import numpy as np
import torch

from depth_policy.collect import Teacher, run_episode
from depth_policy.common import ROOT, atomic_json, load_config, sha256_file, source_identity
from depth_policy.episodes import episode_index
from depth_policy.evaluate import Student
from depth_policy.manifest import load_manifest
from depth_policy.simulation import make_env, observe, task_and_states
from depth_policy.summarize import summarize


def state_hash(state):
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def object_positions(environment):
    return {name: environment.sim.data.get_joint_qpos(obj.joints[-1])[:3].tolist()
            for name, obj in environment.env.objects_dict.items()}


def object_quaternions(environment):
    return {name: environment.sim.data.get_joint_qpos(obj.joints[-1])[3:7].tolist()
            for name, obj in environment.env.objects_dict.items()}


def prepare_bank(config, mode, count, seed):
    if config.get("evaluation_only") and mode != "random":
        raise ValueError("Custom evaluation distribution cannot be labelled official")
    _, official_states, initial_file = task_and_states(config)
    manifest = load_manifest(config, official_states, initial_file)
    if count < 1 or (mode == "official" and count != len(official_states)):
        raise ValueError("Official evaluation must include every official initial state")
    bank = {"schema_version": 1, "mode": mode, "config": config, "seed": seed,
            "source": source_identity(ROOT / "depth_policy"),
            "official_manifest_sha256": sha256_file(config["manifest"]),
            "initial_file_sha256": sha256_file(initial_file), "states": [],
            "distribution": "official saved states" if mode == "official" else
            "fresh native LIBERO resets within unchanged task placement constraints; not whole-table randomization",
            "selection": "fixed seeds before policy evaluation; no filtering by policy outcomes"}
    if config.get("evaluation_only"):
        bank["distribution"] = config["evaluation_distribution"]
    if mode == "official":
        for entry, state in zip(manifest["states"], official_states, strict=True):
            assert state_hash(state) == entry["sha256"]
            bank["states"].append({**entry, "seed": config["seed"] + entry["id"],
                                   "state": state.tolist()})
        return bank
    official_hashes = {entry["sha256"] for entry in manifest["states"]}
    seen = set()
    positions = []
    environment, _ = make_env(config, seed)
    try:
        for index in range(count):
            episode_seed = seed + index
            np.random.seed(episode_seed)
            environment.seed(episode_seed)
            raw = environment.env.reset()
            state = np.asarray(environment.get_sim_state(), dtype=np.float64).copy()
            raw = environment.set_init_state(state)
            digest = state_hash(state)
            if digest in official_hashes or digest in seen or not np.isfinite(state).all():
                raise ValueError("Invalid or duplicate random initial state; stop without resampling")
            seen.add(digest)
            xml_digest = hashlib.sha256(environment.sim.model.get_xml().encode()).hexdigest()
            initial_positions = object_positions(environment)
            initial_quaternions = object_quaternions(environment)
            for name, bounds in config.get("evaluation_center_bounds", {}).items():
                position = np.asarray(initial_positions[name][:2])
                if not np.all((position >= bounds["minimum_xy_m"]) & (position <= bounds["maximum_xy_m"])):
                    raise ValueError("Sampled object center is outside the declared evaluation range")
            observe(environment, raw)
            if environment.check_success():
                raise ValueError("Random reset already satisfies the goal; stop without resampling")
            for step in range(config["settle_steps"]):
                raw, _, _, _ = environment.step([0.0] * 6 + [-1.0])
            observe(environment, raw)
            settled = environment.get_sim_state().copy()
            if not np.isfinite(settled).all() or environment.check_success():
                raise ValueError("Invalid or already-solved settled state; stop without resampling")
            settled_positions = object_positions(environment)
            settled_quaternions = object_quaternions(environment)
            np.random.seed(episode_seed)
            environment.seed(episode_seed)
            environment.env.reset()
            if hashlib.sha256(environment.sim.model.get_xml().encode()).hexdigest() != xml_digest:
                raise ValueError("Seed does not reproduce model XML")
            np.testing.assert_allclose(environment.get_sim_state(), state, atol=1e-12, rtol=0)
            environment.set_init_state(state)
            for step in range(config["settle_steps"]):
                environment.step([0.0] * 6 + [-1.0])
            np.testing.assert_allclose(environment.get_sim_state(), settled, atol=1e-10, rtol=0)
            positions.append(initial_positions)
            bank["states"].append({"id": index, "sha256": digest, "split": "random-test",
                                   "seed": episode_seed, "state": state.tolist(),
                                   "xml_sha256": xml_digest, "initial_positions": initial_positions,
                                   "initial_quaternions": initial_quaternions,
                                   "settled_positions": settled_positions,
                                   "settled_quaternions": settled_quaternions,
                                   "reset_and_settle_reproduced": True})
            print(json.dumps({"prepared": index + 1, "total": count, "seed": episode_seed}), flush=True)
    finally:
        environment.close()
    bank["position_ranges"] = {}
    for name in positions[0]:
        values = np.asarray([record[name] for record in positions])
        if len(np.unique(values, axis=0)) < 2 and count > 1:
            raise ValueError(f"No positional diversity for {name}")
        bank["position_ranges"][name] = {"minimum_xyz_m": values.min(axis=0).tolist(),
                                         "maximum_xyz_m": values.max(axis=0).tolist(),
                                         "span_xyz_m": np.ptp(values, axis=0).tolist()}
    return bank


def check_records(records, bank, policy_metadata, bank_digest, replan_steps, record_images=True):
    expected = {entry["id"]: entry for entry in bank["states"]}
    seen = set()
    for record in records:
        entry = expected.get(record["initial_state_id"])
        if entry is None or record["initial_state_id"] in seen:
            raise ValueError("Unexpected or duplicated evaluation initial state")
        if (record["teacher"] != policy_metadata or record.get("evaluation_bank_sha256") != bank_digest
                or record["seed"] != entry["seed"] or record["initial_state_sha256"] != entry["sha256"]
                or record["split"] != entry["split"]
                or record.get("record_images", True) != record_images
                or record["config"]["teacher_replan_steps"] != replan_steps):
            raise ValueError("Evaluation identity or configuration mismatch")
        if record["termination_reason"] not in {"success", "timeout"}:
            raise ValueError("Interrupted or exceptional rollout requires diagnosis; do not silently retry")
        seen.add(entry["id"])
    return seen


def evaluate_bank(args):
    bank = json.loads(Path(args.bank).read_text())
    digest = sha256_file(args.bank)
    config = dict(bank["config"])
    config["teacher_replan_steps"] = args.replan_steps
    if args.replan_steps < 1 or (args.limit is not None and not 1 <= args.limit <= len(bank["states"])):
        raise ValueError("Invalid evaluation limit or replan interval")
    selected = bank["states"] if args.limit is None else bank["states"][:args.limit]
    torch.set_num_threads(4)
    policy = Teacher(config) if args.teacher else Student(args.checkpoint, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list(output.glob("*.partial.h5")):
            raise ValueError("Incomplete files require diagnosis")
        previous = episode_index(output)
        done = check_records(previous, bank, policy.metadata, digest, args.replan_steps, not args.compact)
        if not done <= {entry["id"] for entry in selected}:
            raise ValueError("Output contains episodes outside this selection")
        stop = {"requested": False}
        def request_stop(signum, frame):
            stop["requested"] = True
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        environment, task = make_env(config, bank["seed"])
        try:
            for entry in selected:
                if stop["requested"]:
                    break
                if entry["id"] in done:
                    continue
                if shutil.disk_usage(output).free < args.min_free_gib * 1024 ** 3:
                    raise RuntimeError("Evaluation stopped before next episode: insufficient free disk space")
                state = np.asarray(entry["state"], dtype=np.float64)
                if state_hash(state) != entry["sha256"]:
                    raise ValueError("Saved initial state hash mismatch")
                metadata = {"task_name": config["task"], "suite": config["suite"],
                            "split": entry["split"], "initial_state_id": entry["id"],
                            "initial_state_sha256": entry["sha256"], "seed": entry["seed"],
                            "teacher": policy.metadata, "depth_units": "m", "config": config,
                            "control_freq": config["control_freq"], "orientation": "flip_both_axes",
                            "manifest_sha256": digest, "evaluation_bank_sha256": digest,
                            "evaluation_mode": bank["mode"], "evaluation_only": True,
                            "record_images": not args.compact,
                            "evaluation_source": source_identity(ROOT / "depth_policy")}
                result = run_episode(environment, task, state, metadata, config, policy, output, stop)
                with h5py.File(result["path"], "r") as stream:
                    np.testing.assert_allclose(stream["initial_sim_state"][:], state, atol=1e-12, rtol=0)
                    if "xml_sha256" in entry:
                        actual_xml = hashlib.sha256(stream["model_xml"].asstr()[()].encode()).hexdigest()
                        if actual_xml != entry["xml_sha256"]:
                            raise ValueError("Executed scene differs from prepared random scene")
        finally:
            environment.close()
        records = episode_index(output)
        done = check_records(records, bank, policy.metadata, digest, args.replan_steps, not args.compact)
        report = summarize(output)
        report.update({"evaluation_bank": str(Path(args.bank).resolve()), "evaluation_bank_sha256": digest,
                       "mode": bank["mode"], "target_episodes": len(selected),
                       "completed": len(done) == len(selected), "policy": policy.metadata,
                       "distribution": bank["distribution"], "evaluation_only": True,
                       "official_split_counts": dict(collections.Counter(entry["split"] for entry in selected))})
        atomic_json(output / "summary.json", report)
        print(json.dumps({key: report[key] for key in ("completed", "target_episodes", "successes", "mode")}), flush=True)
        if not report["completed"]:
            raise RuntimeError("Evaluation stopped before completion")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/wine.json")
    prepare.add_argument("--mode", choices=["official", "random"], required=True)
    prepare.add_argument("--count", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=2026092300)
    prepare.add_argument("--bank", required=True)
    run = commands.add_parser("run")
    run.add_argument("--bank", required=True)
    policies = run.add_mutually_exclusive_group(required=True)
    policies.add_argument("--checkpoint")
    policies.add_argument("--teacher", action="store_true")
    run.add_argument("--device", default="cpu")
    run.add_argument("--replan-steps", type=int, default=4)
    run.add_argument("--limit", type=int)
    run.add_argument("--output", required=True)
    run.add_argument("--compact", action="store_true")
    run.add_argument("--min-free-gib", type=float, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        if Path(args.bank).exists():
            raise FileExistsError("Initial-state bank already exists; do not overwrite")
        bank = prepare_bank(load_config(args.config), args.mode, args.count, args.seed)
        atomic_json(args.bank, bank)
        print(json.dumps({"bank": args.bank, "states": len(bank["states"])}))
    else:
        evaluate_bank(args)


if __name__ == "__main__":
    main()
