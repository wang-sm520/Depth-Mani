"""Collect full teacher demonstrations from distinct native LIBERO resets."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import traceback
from datetime import datetime, timezone

import h5py
import numpy as np

from depth_policy.collect import Teacher, run_episode
from depth_policy.common import ROOT, atomic_json, load_config, sha256_file, source_identity
from depth_policy.episodes import episode_index
from depth_policy.simulation import make_env, task_and_states


TASK = "put_the_cream_cheese_in_the_bowl"
TARGETS = {"train": 300, "validation": 7}
CHEESE = "cream_cheese_1"
BOWL = "akita_black_bowl_1"


def state_hash(state):
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def check_config(config):
    if config["suite"] != "libero_goal" or config["task"] != TASK:
        raise ValueError("Random teacher collection is restricted to the official cream cheese task")
    if config.get("evaluation_only") or "evaluation_bddl" in config:
        raise ValueError("Evaluation-only scenes cannot supply training demonstrations")


def create_bank(config, official_states, initial_file, counts, seeds):
    check_config(config)
    if any(counts[split] < TARGETS[split] for split in TARGETS):
        raise ValueError("Candidate schedule is shorter than its successful demonstration target")
    ranges = {split: list(range(seeds[split], seeds[split] + counts[split])) for split in TARGETS}
    if set(ranges["train"]) & set(ranges["validation"]):
        raise ValueError("Training and validation attempt seeds overlap")
    if any(seed < 0 or seed + counts[split] > 2**32 for split, seed in seeds.items()):
        raise ValueError("Attempt seeds must fit in the NumPy random seed range")
    states = []
    for split in TARGETS:
        states.extend({"id": len(states), "seed": seed, "split": split} for seed in ranges[split])
    return {"schema_version": 1, "task": TASK, "suite": "libero_goal", "config": config,
            "targets": TARGETS, "seed_ranges": {split: {"start": seeds[split], "count": counts[split]}
                                       for split in TARGETS},
            "official_initial_file_sha256": sha256_file(initial_file),
            "official_initial_state_sha256": sorted({state_hash(state) for state in official_states}),
            "states": states}


def load_bank(path, config, official_states, initial_file):
    bank = json.loads(Path(path).read_text())
    expected = create_bank(config, official_states, initial_file,
                           {name: bank["seed_ranges"][name]["count"] for name in TARGETS},
                           {name: bank["seed_ranges"][name]["start"] for name in TARGETS})
    if bank != expected:
        raise ValueError("Frozen teacher attempt schedule does not match this task or official initial states")
    return bank


def object_poses(environment):
    poses = {}
    for name, obj in environment.env.objects_dict.items():
        qpos = np.asarray(environment.sim.data.get_joint_qpos(obj.joints[-1]), dtype=np.float64)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError(f"Object {name} does not have a finite free-joint world pose")
        position, quaternion = qpos[:3], qpos[3:]
        w, x, y, z = quaternion
        poses[name] = {"position_xyz_m": position.tolist(), "quaternion_wxyz": quaternion.tolist(),
                       "yaw_rad": float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))}
    if CHEESE not in poses or BOWL not in poses:
        raise ValueError("Task objects are missing from the native reset")
    return poses


def relative_position(poses):
    return (np.asarray(poses[CHEESE]["position_xyz_m"]) -
            np.asarray(poses[BOWL]["position_xyz_m"])).tolist()


def verify_episode(record, row):
    if record["seed"] != row["seed"] or record["split"] != row["split"]:
        raise ValueError("Teacher attempt seed or split differs from the frozen schedule")
    if (record["initial_state_sha256"] != row["initial_state_sha256"]
            or record["object_initial_poses"] != row["object_initial_poses"]
            or record["relative_cream_cheese_to_bowl_xyz_m"] != row["relative_cream_cheese_to_bowl_xyz_m"]):
        raise ValueError("Teacher trajectory initial state or object poses differ from the attempt ledger")
    with h5py.File(record["path"], "r") as stream:
        if stream.attrs["schema_version"] != 1:
            raise ValueError("Teacher training data must retain full RGB-D frames")
        if state_hash(stream["initial_sim_state"][:]) != row["initial_state_sha256"]:
            raise ValueError("Teacher trajectory does not restore its sampled simulator state")
        if hashlib.sha256(stream["model_xml"].asstr()[()].encode()).hexdigest() != row["xml_sha256"]:
            raise ValueError("Teacher trajectory was recorded in a different model XML")


def reconcile(bank, digest, directory, index_path, recovery_note=None):
    if list(directory.glob("*.partial.h5")):
        raise ValueError("Incomplete teacher trajectories require diagnosis before collection resumes")
    ledger = json.loads(index_path.read_text()) if index_path.exists() else {
        "schema_version": 1, "bank_sha256": digest, "targets": TARGETS, "entries": [], "recoveries": []}
    if {key: ledger[key] for key in ("schema_version", "bank_sha256", "targets")} != {
            "schema_version": 1, "bank_sha256": digest, "targets": TARGETS}:
        raise ValueError("Attempt ledger is not associated with the frozen seed schedule")
    scheduled = {entry["id"]: entry for entry in bank["states"]}
    indexed = {}
    for row in ledger["entries"]:
        entry = scheduled.get(row["id"])
        if entry is None or row["id"] in indexed or any(row[name] != entry[name] for name in ("seed", "split")):
            raise ValueError("Attempt ledger contains a duplicate or foreign seed")
        indexed[row["id"]] = row
    records = {}
    hashes = set()
    for record in episode_index(directory):
        entry = scheduled.get(record.get("initial_state_id"))
        if (entry is None or record["initial_state_id"] in records or
                any(record.get(name) != entry[name] for name in ("seed", "split")) or
                record.get("collection_bank_sha256") != digest or
                record.get("record_images", True) is not True or
                record["teacher"].get("policy_config") != "pi05_libero"):
            raise ValueError("Output contains a duplicate or foreign teacher trajectory")
        records[entry["id"]] = record
    for identifier, record in records.items():
        row = indexed.get(identifier)
        if row is None:
            row = {**scheduled[identifier], "status": "started"}
            ledger["entries"].append(row)
            indexed[identifier] = row
        if row["status"] not in {"started", record["termination_reason"]}:
            raise ValueError("Attempt ledger marks a trajectory for diagnosis; do not overwrite its status")
        for name in ("initial_state_sha256", "object_initial_poses",
                     "relative_cream_cheese_to_bowl_xyz_m", "xml_sha256"):
            if name in row and row[name] != record[name]:
                raise ValueError(f"Attempt ledger differs from trajectory: {name}")
            row[name] = record[name]
        row["status"] = record["termination_reason"]
        row["success"] = record["success"]
        row["path"] = record["path"]
        verify_episode(record, row)
        if row["initial_state_sha256"] in hashes or row["initial_state_sha256"] in bank["official_initial_state_sha256"]:
            raise ValueError("Native reset duplicated a prior or official initial state")
        hashes.add(row["initial_state_sha256"])
    for identifier, row in indexed.items():
        if identifier not in records and row["status"] not in {"exception", "interrupted"}:
            raise ValueError("Attempt started without a completed trajectory; inspect its ledger and partial files")
        if identifier not in records and "initial_state_sha256" in row:
            if row["initial_state_sha256"] in hashes:
                raise ValueError("Two teacher attempts used the same sampled initial state")
            hashes.add(row["initial_state_sha256"])
    exceptional = sorted(identifier for identifier, row in indexed.items()
                         if row["status"] in {"exception", "interrupted"})
    acknowledged = {identifier for recovery in ledger.get("recoveries", [])
                    for identifier in recovery["attempt_ids"]}
    pending = sorted(set(exceptional) - acknowledged)
    if pending and recovery_note:
        ledger.setdefault("recoveries", []).append({"attempt_ids": pending,
                                                    "note": recovery_note,
                                                    "at": datetime.now(timezone.utc).isoformat()})
        pending = []
    atomic_json(index_path, ledger)
    if pending:
        raise RuntimeError(f"Exceptional or interrupted teacher attempts {pending} require diagnosis before resuming")
    return ledger, indexed, hashes


def collect(config, bank_path, split, attempts, output, min_free_gib=10, recovery_note=None):
    if attempts < 1 or split not in TARGETS:
        raise ValueError("Choose a positive number of attempts and a valid split")
    check_config(config)
    _, official_states, initial_file = task_and_states(config)
    bank = load_bank(bank_path, config, official_states, initial_file)
    digest = sha256_file(bank_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index_path = output / "attempt-index.json"
        ledger, completed, hashes = reconcile(bank, digest, output, index_path, recovery_note)
        successes = sum(row["status"] == "success" for row in completed.values() if row["split"] == split)
        if successes >= TARGETS[split]:
            return {"split": split, "successes": successes, "target": TARGETS[split], "attempts_this_run": 0}
        candidates = [entry for entry in bank["states"] if entry["split"] == split and entry["id"] not in completed]
        if not candidates:
            raise RuntimeError(f"Exhausted the frozen {split} attempt schedule without reaching its target")
        policy = Teacher(config)
        previous = episode_index(output)
        if any(record["teacher"] != policy.metadata for record in previous):
            raise ValueError("Existing teacher trajectories belong to a different checkpoint or backend")
        stop = {"requested": False}

        def request_stop(signum, frame):
            stop["requested"] = True

        old_term = signal.signal(signal.SIGTERM, request_stop)
        old_int = signal.signal(signal.SIGINT, request_stop)
        environment = None
        processed = 0
        try:
            environment, task = make_env(config, bank["states"][0]["seed"])
            provenance = {
                "task_name": config["task"], "suite": config["suite"], "teacher": policy.metadata,
                "depth_units": "m", "orientation": "flip_both_axes", "control_freq": config["control_freq"],
                "resolution": config["resolution"], "config": config,
                "collection_bank_sha256": digest, "manifest_sha256": digest,
                "record_images": True,
                "state_convention": "eef_xyz_m + eef_axisangle_rad + two_gripper_joint_positions_m",
                "action_convention": "OSC_POSE input: delta_xyz + delta_axisangle + gripper; controller scales inputs",
                "runtime_environment": {name: os.environ.get(name) for name in
                                        ("MUJOCO_GL", "PYOPENGL_PLATFORM", "NUMBA_DISABLE_JIT", "CUDA_VISIBLE_DEVICES")},
                "collector_source": source_identity(ROOT / "depth_policy"),
                "openpi_source": source_identity(config["openpi_root"]),
                "versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "torch", "robosuite", "mujoco", "h5py")},
            }
            for entry in candidates:
                if processed >= attempts or successes >= TARGETS[split] or stop["requested"]:
                    break
                if shutil.disk_usage(output).free < min_free_gib * 1024 ** 3:
                    raise RuntimeError("Require free disk space before the next full RGB-D demonstration")
                row = {**entry, "status": "started"}
                ledger["entries"].append(row)
                atomic_json(index_path, ledger)
                processed += 1
                try:
                    np.random.seed(entry["seed"])
                    environment.seed(entry["seed"])
                    environment.reset()
                    state = np.asarray(environment.get_sim_state(), dtype=np.float64).copy()
                    state_digest = state_hash(state)
                    if (not np.isfinite(state).all() or state_digest in hashes or
                            state_digest in bank["official_initial_state_sha256"]):
                        raise ValueError("Invalid, repeated, or official native-reset initial state")
                    poses = object_poses(environment)
                    row.update({"initial_state_sha256": state_digest, "initial_sim_state": state.tolist(),
                                "xml_sha256": hashlib.sha256(environment.sim.model.get_xml().encode()).hexdigest(),
                                "object_initial_poses": poses,
                                "relative_cream_cheese_to_bowl_xyz_m": relative_position(poses)})
                    if environment.check_success():
                        raise ValueError("Native reset already satisfies the task goal")
                    atomic_json(index_path, ledger)
                    metadata = {**provenance, **entry, "initial_state_id": entry["id"],
                                "initial_state_sha256": state_digest,
                                "xml_sha256": row["xml_sha256"], "object_initial_poses": poses,
                                "relative_cream_cheese_to_bowl_xyz_m": row["relative_cream_cheese_to_bowl_xyz_m"]}
                    result = run_episode(environment, task, state, metadata, config, policy, output, stop)
                    records = [record for record in episode_index(output)
                               if record["initial_state_id"] == entry["id"]]
                    if len(records) != 1:
                        raise ValueError("Exactly one recorded trajectory is required per attempt seed")
                    verify_episode(records[0], row)
                    row.update({"status": result["reason"], "success": result["success"], "path": result["path"]})
                    hashes.add(state_digest)
                    if result["reason"] in {"interrupted", "exception"}:
                        raise RuntimeError(f"Teacher attempt {entry['id']} ended in {result['reason']}")
                    successes += int(result["success"])
                    atomic_json(index_path, ledger)
                except BaseException:
                    records = [record for record in episode_index(output)
                               if record["initial_state_id"] == entry["id"]]
                    if len(records) == 1:
                        row["path"] = records[0]["path"]
                        row["success"] = records[0]["success"]
                    row["status"] = "interrupted" if stop["requested"] else "exception"
                    row["error"] = traceback.format_exc()
                    atomic_json(index_path, ledger)
                    raise
        finally:
            if environment is not None:
                environment.close()
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        if successes < TARGETS[split] and processed < attempts and len(candidates) <= processed:
            raise RuntimeError(f"Exhausted the frozen {split} attempt schedule without reaching its target")
        return {"split": split, "successes": successes, "target": TARGETS[split],
                "attempts_this_run": processed, "bank_sha256": digest}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--bank", required=True)
    prepare.add_argument("--train-attempts", type=int, required=True)
    prepare.add_argument("--validation-attempts", type=int, required=True)
    prepare.add_argument("--train-seed", type=int, required=True)
    prepare.add_argument("--validation-seed", type=int, required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--bank", required=True)
    run.add_argument("--split", choices=tuple(TARGETS), required=True)
    run.add_argument("--attempts", type=int, required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--min-free-gib", type=float, default=10)
    run.add_argument("--recovery-note", help="Document diagnosed prior exceptions before continuing with new seeds")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        check_config(config)
        path = Path(args.bank)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.exists():
                raise FileExistsError("Attempt seed schedule already exists; refusing to replace it")
            _, states, initial_file = task_and_states(config)
            bank = create_bank(config, states, initial_file,
                               {"train": args.train_attempts, "validation": args.validation_attempts},
                               {"train": args.train_seed, "validation": args.validation_seed})
            atomic_json(path, bank)
        print(json.dumps({"bank": str(path), "counts": bank["seed_ranges"], "targets": TARGETS}), flush=True)
    else:
        result = collect(config, args.bank, args.split, args.attempts, args.output,
                         args.min_free_gib, args.recovery_note)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
