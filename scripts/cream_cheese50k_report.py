"""Audit and report the frozen cream-cheese imitation-learning experiment."""

import argparse
import collections
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

from depth_policy.common import ROOT, atomic_json, sha256_file, source_identity
from depth_policy.data import (
    build_vocabulary, data_fingerprint, fit_statistics, training_episodes,
)
from depth_policy.episodes import episode_index
from depth_policy.evaluate import Student
from depth_policy.evaluate_suite import check_records, state_hash
from depth_policy.validate import validate_episode

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


CHEESE = "cream_cheese_1"
BOWL = "akita_black_bowl_1"
COLORS = {"depth": "#2472b4", "rgb": "#d97724"}
GROUP_TITLES = {"official-range100": "Original placement (100 states)",
                "expanded3x200": "Cheese x/y width 3x (200 states)"}


def require_official_instruction(actual, config):
    if actual != config["official_instruction"]:
        raise ValueError("Episode did not use the official BDDL instruction")


def group_count(config, group):
    return config["groups"][group]["count"]


def _digest_xml(stream):
    return hashlib.sha256(stream["model_xml"].asstr()[()].encode()).hexdigest()


def _pose(entry, name):
    poses = entry["object_initial_poses"]
    pose = poses[name]
    position = np.asarray(pose["position_xyz_m"], dtype=float)
    quaternion = np.asarray(pose["quaternion_wxyz"], dtype=float)
    yaw = float(pose["yaw_rad"])
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all():
        raise ValueError(f"Invalid object pose: {name}")
    if not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-5):
        raise ValueError(f"Invalid quaternion: {name}")
    if not np.isfinite(yaw):
        raise ValueError(f"Invalid yaw: {name}")
    w, x, y, z = quaternion
    expected_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if abs((yaw - expected_yaw + np.pi) % (2 * np.pi) - np.pi) > 1e-6:
        raise ValueError(f"Yaw disagrees with object quaternion: {name}")
    return position, quaternion, yaw


def validate_attempt_ledger(schedule, ledger):
    """Check the frozen seed schedule, including failed and exceptional attempts."""
    states = schedule["states"]
    if schedule["schema_version"] != 1 or ledger["schema_version"] != 1:
        raise ValueError("Unknown collection bank or attempt ledger schema")
    if [state["id"] for state in states] != list(range(len(states))):
        raise ValueError("Schedule IDs must be stable dense indexes")
    if len({state["seed"] for state in states}) != len(states):
        raise ValueError("Schedule reuses a random seed")
    entries = ledger["entries"]
    if not entries:
        raise ValueError("No recorded collection attempts")
    by_id = {}
    hashes = set()
    paths = set()
    object_names = None
    for entry in entries:
        identifier = entry["id"]
        if identifier in by_id or identifier not in range(len(states)):
            raise ValueError("Duplicated or unexpected collection attempt")
        expected = states[identifier]
        if (entry["seed"], entry["split"]) != (expected["seed"], expected["split"]):
            raise ValueError("Attempt differs from frozen seed schedule")
        status = entry["status"]
        if status not in {"success", "timeout", "environment_done_without_success", "exception", "interrupted"}:
            raise ValueError(f"Unexplained collection status: {status}")
        if status in {"exception", "interrupted"} and not entry.get("error"):
            raise ValueError("Exceptional attempt lacks a diagnostic error")
        if entry.get("path"):
            if entry["path"] in paths:
                raise ValueError("Two attempts refer to one trajectory")
            paths.add(entry["path"])
        elif status not in {"exception", "interrupted"}:
            raise ValueError("A completed attempt has no recorded trajectory")
        if entry.get("initial_state_sha256"):
            digest = entry["initial_state_sha256"]
            if digest in hashes or state_hash(np.asarray(entry["initial_sim_state"], dtype=np.float64)) != digest:
                raise ValueError("Duplicate or inconsistent initial simulator state")
            if digest in schedule["official_initial_state_sha256"]:
                raise ValueError("Training initial state reuses an official saved state")
            hashes.add(digest)
            poses = entry["object_initial_poses"]
            if object_names is None:
                object_names = set(poses)
            if set(poses) != object_names or not {CHEESE, BOWL} <= object_names:
                raise ValueError("Missing or inconsistent randomized object pose metadata")
            for name in object_names:
                _pose(entry, name)
            difference = _pose(entry, CHEESE)[0] - _pose(entry, BOWL)[0]
            np.testing.assert_allclose(difference, entry["relative_cream_cheese_to_bowl_xyz_m"], atol=1e-9)
            if not entry.get("xml_sha256"):
                raise ValueError("Scene XML identity missing")
        elif status not in {"exception", "interrupted"}:
            raise ValueError("Completed attempt lacks a reset state")
        by_id[identifier] = entry
    for split in ("train", "validation"):
        scheduled = [state["id"] for state in states if state["split"] == split]
        attempted = sorted(identifier for identifier in by_id if by_id[identifier]["split"] == split)
        if attempted != scheduled[:len(attempted)]:
            raise ValueError(f"{split} attempts skipped a scheduled seed")
    return by_id


def coverage_summary(entries, bins=6):
    """World-frame positions, yaw, and target-relative coverage, all attempts vs successes."""
    sampled = [item for item in entries if item.get("initial_state_sha256")]
    successes = [item for item in sampled if item["status"] == "success"]
    if not sampled or not successes:
        raise ValueError("Insufficient sampled states for coverage comparison")
    result = {}
    for name in [*sorted(sampled[0]["object_initial_poses"]), "cheese_minus_bowl"]:
        def values(selection):
            if name == "cheese_minus_bowl":
                return np.asarray([item["relative_cream_cheese_to_bowl_xyz_m"][:2] for item in selection])
            return np.asarray([_pose(item, name)[0][:2] for item in selection])

        all_xy, success_xy = values(sampled), values(successes)
        lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
        edges = [np.linspace(lower[axis] - 1e-8, upper[axis] + 1e-8, bins + 1) for axis in (0, 1)]
        attempted_grid = np.histogram2d(all_xy[:, 0], all_xy[:, 1], bins=edges)[0].astype(int)
        success_grid = np.histogram2d(success_xy[:, 0], success_xy[:, 1], bins=edges)[0].astype(int)
        coverage = {"attempted_xy_min_m": lower.tolist(), "attempted_xy_max_m": upper.tolist(),
                    "successful_xy_min_m": success_xy.min(axis=0).tolist(),
                    "successful_xy_max_m": success_xy.max(axis=0).tolist(),
                    "grid_x_edges_m": edges[0].tolist(), "grid_y_edges_m": edges[1].tolist(),
                    "attempted_grid": attempted_grid.tolist(), "successful_grid": success_grid.tolist(),
                    "attempted_occupied_cells": int(np.count_nonzero(attempted_grid)),
                    "successful_occupied_cells": int(np.count_nonzero(success_grid))}
        if name != "cheese_minus_bowl":
            yaw_all = np.asarray([_pose(item, name)[2] for item in sampled])
            yaw_success = np.asarray([_pose(item, name)[2] for item in successes])
            coverage["attempted_yaw_range_rad"] = [float(yaw_all.min()), float(yaw_all.max())]
            coverage["successful_yaw_range_rad"] = [float(yaw_success.min()), float(yaw_success.max())]
        result[name] = coverage
    return result


def audit_collector_sources(records, current_source):
    """Allow the documented replay verifier update, never an unrecorded collection change."""
    training = [item for item in records if item["split"] == "train"]
    if not training:
        raise ValueError("No training source identity to audit")
    baseline = min(training, key=lambda item: item["initial_state_id"]).get("collector_source")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("Training trajectory lacks collector source hashes")

    def digest(source):
        return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()

    def changes(source):
        if not isinstance(source, dict) or set(source) != set(baseline):
            raise ValueError("Collector source file list changed")
        different = sorted(name for name in baseline if source[name] != baseline[name])
        if set(different) - {"replay.py"}:
            raise ValueError(f"Collection implementation changed: {different}")
        return different

    versions = {}
    for record in records:
        source = record.get("collector_source")
        different = changes(source)
        version = versions.setdefault(digest(source), {
            "source_sha256": digest(source), "files": source,
            "changed_from_training_baseline": different, "attempt_ids": [],
            "splits": collections.Counter()})
        version["attempt_ids"].append(record["initial_state_id"])
        version["splits"][record["split"]] += 1
    current_changes = changes(current_source)
    return {"training_baseline_sha256": digest(baseline),
            "current_sha256": digest(current_source),
            "current_changes_from_training_baseline": current_changes,
            "permitted_audit_only_change": "replay.py",
            "versions": [
                {**version, "attempt_ids": sorted(version["attempt_ids"]),
                 "splits": dict(version["splits"])}
                for _, version in sorted(versions.items())]}


def audit_collection(config):
    directory = ROOT / config["training_data"]
    schedule_path = ROOT / config["attempt_bank"]
    ledger_path = directory / "attempt-index.json"
    schedule = json.loads(schedule_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    digest = sha256_file(schedule_path)
    if schedule["config"]["official_instruction"] != config["official_instruction"]:
        raise ValueError("Teacher schedule uses the suite-derived or a different instruction")
    if ledger["bank_sha256"] != digest:
        raise ValueError("Collection ledger references a different seed schedule")
    by_id = validate_attempt_ledger(schedule, ledger)
    exceptional = {identifier for identifier, entry in by_id.items()
                   if entry["status"] in {"exception", "interrupted"}}
    recoveries = ledger.get("recoveries", [])
    acknowledged = set()
    for recovery in recoveries:
        ids = recovery["attempt_ids"]
        if (not ids or not set(ids) <= exceptional or set(ids) & acknowledged
                or not recovery.get("note") or not recovery.get("at")):
            raise ValueError("Collection recovery entry is incomplete or refers to a nonexceptional attempt")
        acknowledged.update(ids)
    if exceptional != acknowledged:
        raise ValueError("Unacknowledged exceptional teacher attempts need diagnosis")
    manifest = {"states": [{"split": state["split"],
                           "sha256": by_id.get(state["id"], {}).get("initial_state_sha256", "")}
                          for state in schedule["states"]]}
    records = episode_index(directory)
    if list(directory.glob("*.partial.h5")):
        raise ValueError("Unfinished teacher HDF5 files need diagnosis")
    collector_sources = audit_collector_sources(records, source_identity(ROOT / "depth_policy"))
    recorded = {str(Path(record["path"]).resolve()): record for record in records}
    expected = {str(Path(entry["path"]).resolve()) for entry in by_id.values() if entry.get("path")}
    if set(recorded) != expected:
        raise ValueError("Unindexed, missing, or duplicate teacher trajectories")
    for entry in by_id.values():
        if not entry.get("path"):
            continue
        path = str(Path(entry["path"]).resolve())
        record = recorded[path]
        require_official_instruction(record.get("instruction"), config)
        if (record["initial_state_id"] != entry["id"] or record["seed"] != entry["seed"]
                or record["initial_state_sha256"] != entry["initial_state_sha256"]
                or record["split"] != entry["split"] or record["success"] != (entry["status"] == "success")
                or record["termination_reason"] != entry["status"]
                or record["collection_bank_sha256"] != digest
                or record["object_initial_poses"] != entry["object_initial_poses"]
                or record["relative_cream_cheese_to_bowl_xyz_m"] != entry["relative_cream_cheese_to_bowl_xyz_m"]):
            raise ValueError("Teacher trajectory and attempt ledger disagree")
        if (record["teacher"].get("policy_config") != "pi05_libero" or record.get("evaluation_only")
                or Path(record["teacher"].get("checkpoint_path", "")).resolve()
                != Path(schedule["config"]["teacher_checkpoint"]).resolve()
                or record["config"] != schedule["config"]):
            raise ValueError("Teacher identity, BDDL instruction, or recording configuration mismatch")
        with h5py.File(path, "r") as stream:
            if (stream.attrs["schema_version"] != 1 or _digest_xml(stream) != entry["xml_sha256"]
                    or state_hash(stream["initial_sim_state"][:]) != entry["initial_state_sha256"]):
                raise ValueError("Full-image teacher episode or initial scene identity invalid")
        validate_episode(path, manifest)
    selected = {}
    all_hashes = {entry["initial_state_sha256"] for entry in by_id.values()
                  if entry.get("initial_state_sha256")}
    for split, target in (("train", 300), ("validation", 7)):
        selected[split] = training_episodes(directory, split, target if split == "train" else None)
        successes = [entry for entry in by_id.values() if entry["split"] == split and entry["status"] == "success"]
        if len(selected[split]) != target or len(successes) != target:
            raise ValueError(f"{split} does not contain exactly {target} successful demonstrations")
        if {item["initial_state_id"] for item in selected[split]} != {item["id"] for item in successes}:
            raise ValueError("Training selector and attempt ledger disagree")
        if len({item["initial_state_sha256"] for item in selected[split]}) != target:
            raise ValueError("Successful demonstrations repeat initial states")
    if {item["initial_state_sha256"] for item in selected["train"]} & {
            item["initial_state_sha256"] for item in selected["validation"]}:
        raise ValueError("Train and validation initial states overlap")
    counts = {split: {"attempts": sum(entry["split"] == split for entry in by_id.values()),
                      "successes": len(selected[split]),
                      "termination_counts": dict(collections.Counter(entry["status"] for entry in by_id.values()
                                                                        if entry["split"] == split))}
              for split in ("train", "validation")}
    for split in counts:
        counts[split]["success_rate"] = counts[split]["successes"] / counts[split]["attempts"]
    audit = {"schedule_sha256": digest, "ledger_sha256": sha256_file(ledger_path),
             "recorded_attempts": len(by_id), "splits": counts,
             "collector_source_audit": collector_sources,
             "official_instruction": config["official_instruction"],
             "instruction_source": "BDDL (:language), not suite task-name-derived language",
             "exception_recoveries": recoveries,
             "all_sampled_initial_state_sha256": sorted(all_hashes),
             "selected_data_fingerprints": {split: data_fingerprint(items) for split, items in selected.items()},
             "coverage": {split: coverage_summary([entry for entry in by_id.values() if entry["split"] == split])
                          for split in ("train", "validation")}}
    atomic_json(ROOT / "reports" / f"{config['label']}-collection-audit.json", audit)
    return audit, selected, by_id


def training_metrics(directory, steps):
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if not records or records[-1]["step"] != steps:
        raise ValueError("Training metrics do not reach the final optimizer step")
    logged_steps = [row["step"] for row in records]
    if sorted(set(logged_steps)) != logged_steps:
        raise ValueError("Duplicated or unordered training metrics")
    validation = [row for row in records if "validation_loss" in row]
    if [row["step"] for row in validation] != list(range(500, steps + 1, 500)):
        raise ValueError("Missing or extra 500-step validation losses")
    if not all(np.isfinite(row["train_loss"]) and row["train_loss"] >= 0 for row in records):
        raise ValueError("Nonfinite training loss")
    if not all(np.isfinite(row["validation_loss"]) and row["validation_loss"] >= 0 for row in validation):
        raise ValueError("Nonfinite validation loss")
    return records


def verify_training(config, root, collection):
    fingerprint = collection["selected_data_fingerprints"]
    train = training_episodes(ROOT / config["training_data"], "train", config["training_limit"])
    if data_fingerprint(train) != fingerprint["train"]:
        raise ValueError("Collection changed after freezing training subset")
    expected_statistics = fit_statistics(train)
    expected_vocabulary = build_vocabulary(train)
    reports = {}
    for modality in config["modalities"]:
        directory = root / "training" / modality
        run = json.loads((directory / "run.json").read_text())
        args = run["config"]
        expected = {"modality": modality, "steps": 50000, "checkpoint_every": 10000,
                    "seed": config["training_seed"], "batch_size": 64, "learning_rate": 0.0003,
                    "horizon": 8, "image_size": 128, "depth_max": 3.0, "limit": 300}
        if (any(args.get(key) != value for key, value in expected.items())
                or args["resume"] or args["data"] != config["training_data"]
                or args["save_every"] != 500 or run["initialization"] != "all_student_parameters_random"
                or run["train_episodes"] != 300 or run["train_initial_states"] != 300
                or run["validation_episodes"] != 7):
            raise ValueError(f"Wrong training protocol for {modality}")
        if run["data"] != fingerprint or run["statistics"] != expected_statistics or run["vocabulary"] != expected_vocabulary:
            raise ValueError("Training data or training-only normalization/vocabulary mismatch")
        metrics = training_metrics(directory, config["training_steps"])
        reports[modality] = {"parameters": run["parameters"], "train_frames": run["train_frames"],
                             "validation_frames": run["validation_frames"],
                             "metrics": str(directory / "metrics.jsonl"), "checkpoints": {}}
        for step in config["checkpoint_steps"]:
            path = directory / f"step_{step:06d}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if (checkpoint["step"] != step or checkpoint["config"] != args
                    or checkpoint["data"] != fingerprint or checkpoint["statistics"] != expected_statistics
                    or checkpoint["vocabulary"] != expected_vocabulary
                    or sum(value.numel() for value in checkpoint["model"].values()) != run["parameters"]):
                raise ValueError(f"Incorrect checkpoint identity: {path}")
            reports[modality]["checkpoints"][str(step)] = {"path": str(path), "sha256": sha256_file(path),
                                                             "step": step}
        latest = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
        if latest["step"] != 50000 or metrics[-1]["step"] != latest["step"]:
            raise ValueError("Training did not reach 50000 optimizer steps")
    result = {"training_data_unchanged": True, "train_episodes": 300, "train_initial_states": 300,
              "validation_episodes": 7, "training_data_fingerprint": fingerprint,
              "training_only_statistics_verified": True, "random_initialization": True,
              "modalities": reports}
    atomic_json(ROOT / "reports" / f"{config['label']}-training-audit.json", result)
    return result


def replay_episode(path, report, log):
    with log.open("w") as output:
        subprocess.run(["bash", "scripts/sim_cpu.sh", "-m", "depth_policy.replay",
                        str(path), "--report", str(report)], cwd=ROOT,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    result = json.loads(report.read_text())
    if not result["passed"]:
        raise ValueError(f"Precise simulation replay failed: {path}")
    return result


def audit_teacher_replays(config, selected, collection):
    destination = ROOT / "reports" / config["label"]
    destination.mkdir(parents=True, exist_ok=True)
    replays = {}
    for split in ("train", "validation"):
        episode = min(selected[split], key=lambda item: item["initial_state_id"])
        report = destination / f"teacher-{split}-replay.json"
        replays[split] = replay_episode(episode["path"], report,
                                        destination / f"teacher-{split}-replay.log")
        if not replays[split]["full_rgbd_verified"]:
            raise ValueError(f"Teacher {split} replay did not verify full RGB-D frames")
    collection["teacher_fixed_replay_audit"] = replays
    atomic_json(ROOT / "reports" / f"{config['label']}-collection-audit.json", collection)
    return replays


def _center_bounds(bank):
    config = bank["config"]
    return config.get("evaluation_center_bounds", config["official_center_bounds"])[CHEESE]


def validate_evaluation_banks(config, collection):
    seen = set(collection["all_sampled_initial_state_sha256"])
    schedule = json.loads((ROOT / config["attempt_bank"]).read_text())
    seeds = {entry["seed"] for entry in schedule["states"]}
    banks = {}
    for group, settings in config["groups"].items():
        path = ROOT / "data/cream_cheese/evaluation" / config["label"] / f"{group}.json"
        bank = json.loads(path.read_text())
        if (bank["mode"] != "random" or bank["seed"] != settings["seed"]
                or len(bank["states"]) != settings["count"]
                or bank["config"]["official_instruction"] != config["official_instruction"]
                or bank["config"] != json.loads((ROOT / settings["config"]).read_text())):
            raise ValueError(f"Evaluation bank identity or size differs: {group}")
        bounds = _center_bounds(bank)
        for number, entry in enumerate(bank["states"]):
            if (entry["id"] != number or entry["split"] != "random-test"
                    or entry["sha256"] in seen or entry["seed"] in seeds
                    or state_hash(np.asarray(entry["state"], dtype=np.float64)) != entry["sha256"]
                    or not entry["reset_and_settle_reproduced"] or not entry.get("xml_sha256")):
                raise ValueError("Duplicated, leaked, or nonreproducible evaluation initial state")
            if entry["seed"] != settings["seed"] + number:
                raise ValueError("Frozen evaluation seed schedule mismatch")
            if set(entry["initial_positions"]) != set(entry["initial_quaternions"]):
                raise ValueError("Evaluation bank omits a randomized object orientation")
            for name, quaternion in entry["initial_quaternions"].items():
                if (len(quaternion) != 4 or not np.isfinite(quaternion).all()
                        or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-5)
                        or len(entry["initial_positions"][name]) != 3):
                    raise ValueError("Invalid evaluation object pose")
            if not {CHEESE, BOWL} <= set(entry["initial_positions"]):
                raise ValueError("Cheese or bowl absent from evaluation bank")
            xy = np.asarray(entry["initial_positions"][CHEESE][:2])
            if np.any(xy < bounds["minimum_xy_m"]) or np.any(xy > bounds["maximum_xy_m"]):
                raise ValueError("Cheese sampled outside declared effective center bounds")
            seen.add(entry["sha256"])
            seeds.add(entry["seed"])
        banks[group] = {"path": path, "bank": bank, "sha256": sha256_file(path)}
    original = banks["official-range100"]["bank"]["config"]["official_center_bounds"][CHEESE]
    expanded = banks["expanded3x200"]["bank"]["config"]["evaluation_center_bounds"][CHEESE]
    old_min, old_max = np.asarray(original["minimum_xy_m"]), np.asarray(original["maximum_xy_m"])
    new_min, new_max = np.asarray(expanded["minimum_xy_m"]), np.asarray(expanded["maximum_xy_m"])
    np.testing.assert_allclose(new_max - new_min, 3 * (old_max - old_min), atol=1e-9)
    np.testing.assert_allclose(new_max + new_min, old_max + old_min, atol=1e-9)
    if sum(settings["count"] for settings in config["groups"].values()) != 300:
        raise ValueError("Evaluation protocol must freeze 100 + 200 fresh initial states")
    return banks


def evaluation_coverage(banks):
    result = {}
    for group, info in banks.items():
        bank = info["bank"]
        positions = bank["states"]
        names = sorted(positions[0]["initial_positions"])
        by_object = {}
        for name in names:
            xyz = np.asarray([entry["initial_positions"][name] for entry in positions])
            quaternion = np.asarray([entry["initial_quaternions"][name] for entry in positions])
            w, x, y, z = quaternion.T
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            by_object[name] = {"minimum_xyz_m": xyz.min(axis=0).tolist(),
                               "maximum_xyz_m": xyz.max(axis=0).tolist(),
                               "yaw_range_rad": [float(yaw.min()), float(yaw.max())]}
        xy = np.asarray([_map_xy(entry, "relative-bowl") for entry in positions])
        result[group] = {"sampled_object_ranges": by_object,
                         "cheese_minus_bowl_minimum_xy_m": xy.min(axis=0).tolist(),
                         "cheese_minus_bowl_maximum_xy_m": xy.max(axis=0).tolist(),
                         "declared_candidate_center_bounds_m": _center_bounds(bank),
                         "candidate_support_not_uniform_after_collision_rejection": True}
    return result


def audit_evaluations(config, root, training, banks):
    rows, details, pairs = [], {}, {}
    directory = ROOT / "reports" / config["label"]
    directory.mkdir(parents=True, exist_ok=True)
    for group, bank_info in banks.items():
        bank, digest = bank_info["bank"], bank_info["sha256"]
        count = group_count(config, group)
        for step in config["checkpoint_steps"]:
            paired = {}
            for modality in config["modalities"]:
                key = f"{modality}-{step}-{group}"
                checkpoint = training["modalities"][modality]["checkpoints"][str(step)]
                if sha256_file(checkpoint["path"]) != checkpoint["sha256"]:
                    raise ValueError("Checkpoint changed after training audit")
                policy = Student(checkpoint["path"], "cpu")
                output = root / "evaluation" / f"{modality}-{step}" / group
                summary = json.loads((output / "summary.json").read_text())
                records = episode_index(output)
                if list(output.glob("*.partial.h5")) or not summary["completed"]:
                    raise ValueError(f"Incomplete evaluation group: {key}")
                if (summary["completed_attempts"] != count or summary["policy"] != policy.metadata
                        or summary["evaluation_bank_sha256"] != digest
                        or len(records) != count
                        or len(check_records(records, bank, policy.metadata, digest,
                                              config["replan_steps"], record_images=False)) != count):
                    raise ValueError(f"Evaluation group has missing or mismatched trials: {key}")
                indexed = {record["initial_state_id"]: record for record in records}
                trajectory_audits = []
                for entry in bank["states"]:
                    record = indexed[entry["id"]]
                    require_official_instruction(record.get("instruction"), config)
                    if record["config"] != {**bank["config"], "teacher_replan_steps": config["replan_steps"]}:
                        raise ValueError("Evaluation controller, BDDL instruction, or environment mismatch")
                    with h5py.File(record["path"], "r") as stream:
                        if stream.attrs["error"] or stream.attrs["schema_version"] != 2:
                            raise ValueError("Exceptional or full-image evaluation episode")
                        np.testing.assert_array_equal(stream["initial_sim_state"][:], entry["state"])
                        if _digest_xml(stream) != entry["xml_sha256"]:
                            raise ValueError("Replayed scene XML differs from frozen evaluation bank")
                        steps = stream["steps"]
                        if len(steps["phase"]) > 310 or (not record["success"] and len(steps["phase"]) != 310):
                            raise ValueError("Evaluation outcome is neither task success nor full-budget timeout")
                        first = stream["first_policy_observation"]
                        first_index = int(first.attrs["step_index"])
                        observation = {name: first[name][:] for name in
                                       ("state", f"{modality}_agentview", f"{modality}_wrist")}
                        action = policy.infer(observation, record["instruction"])
                        actions = min(config["replan_steps"], len(steps["phase"]) - first_index)
                        if actions < 1:
                            raise ValueError("Evaluation episode has no policy-controlled action")
                        np.testing.assert_allclose(action[:actions],
                                                   steps["executed_action"][first_index:first_index + actions],
                                                   atol=1e-6, rtol=0)
                    trajectory_audits.append(validate_episode(record["path"], bank))
                outcomes = np.asarray([indexed[entry["id"]]["success"] for entry in bank["states"]], dtype=bool)
                paired[modality] = outcomes
                successes = int(outcomes.sum())
                if successes != summary["successes"]:
                    raise ValueError("Outcome tally disagrees with evaluation summary")
                row = {"distribution": group, "modality": modality, "training_step": step,
                       "attempts": count, "successes": successes, "timeouts": count - successes,
                       "success_rate": successes / count, "checkpoint_sha256": checkpoint["sha256"]}
                rows.append(row)
                report = directory / f"{key}-replay.json"
                replay = replay_episode(indexed[0]["path"], report, directory / f"{key}-replay.log")
                if not replay["image_hashes_verified"]:
                    raise ValueError("Compact replay did not verify full RGB-D frame hashes")
                details[key] = {**row, "bank_sha256": digest, "first_action_chunks_verified": True,
                                "termination_counts": dict(collections.Counter(
                                    record["termination_reason"] for record in records)),
                                "success_by_initial_state": outcomes.tolist(),
                                "trajectory_audit": trajectory_audits, "fixed_id0_replay_audit": replay}
                atomic_json(directory / f"{key}-audit.json", details[key])
                print(json.dumps(row), flush=True)
            pairs[f"{group}-{step}"] = {
                "both_success": int((paired["depth"] & paired["rgb"]).sum()),
                "depth_only_success": int((paired["depth"] & ~paired["rgb"]).sum()),
                "rgb_only_success": int((~paired["depth"] & paired["rgb"]).sum()),
                "both_fail": int((~paired["depth"] & ~paired["rgb"]).sum()),
            }
    return rows, details, pairs


def _limits(xy, padding=0.07):
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    half = max(float(np.max(maximum - minimum)) / 2, 0.005) * (1 + padding)
    center = (minimum + maximum) / 2
    return [(center[index] - half, center[index] + half) for index in (0, 1)]


def make_coverage_figures(config, attempts, coverage, destination):
    for split in ("train", "validation"):
        sampled = [entry for entry in attempts.values()
                   if entry["split"] == split and entry.get("initial_state_sha256")]
        successful = [entry for entry in sampled if entry["status"] == "success"]
        for name, summary in coverage[split].items():
            def xy(items):
                if name == "cheese_minus_bowl":
                    return np.asarray([entry["relative_cream_cheese_to_bowl_xyz_m"][:2] for entry in items])
                return np.asarray([_pose(entry, name)[0][:2] for entry in items])

            all_xy, success_xy = xy(sampled), xy(successful)
            figure, axes = plt.subplots(2, 3, figsize=(15, 9))
            axes[0, 0].scatter(all_xy[:, 0], all_xy[:, 1], s=16, alpha=0.35, color="#808d97",
                               label=f"All sampled attempts ({len(sampled)})")
            axes[0, 0].scatter(success_xy[:, 0], success_xy[:, 1], s=20, alpha=0.68,
                               color="#1d875a", label=f"Successes ({len(successful)})")
            axes[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="Initial x/y; world frame" if name !=
                           "cheese_minus_bowl" else "Cheese minus bowl x/y (m)")
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(alpha=0.15)
            for axis, index in ((axes[0, 1], 0), (axes[0, 2], 1)):
                edges = np.histogram_bin_edges(all_xy[:, index], bins=20)
                axis.hist(all_xy[:, index], bins=edges, alpha=0.55, color="#808d97", label="All")
                axis.hist(success_xy[:, index], bins=edges, alpha=0.55, color="#1d875a", label="Success")
                axis.set(xlabel=f"{'x' if index == 0 else 'y'} (m)", ylabel="Count",
                         title=f"Initial {'x' if index == 0 else 'y'} marginal")
                axis.legend(fontsize=8)
            if name == "cheese_minus_bowl":
                axes[1, 0].axis("off")
                axes[1, 0].text(0.05, 0.8, "Relative target geometry: cheese position - bowl position",
                                transform=axes[1, 0].transAxes, wrap=True)
            else:
                all_yaw = [_pose(entry, name)[2] for entry in sampled]
                success_yaw = [_pose(entry, name)[2] for entry in successful]
                bins = np.linspace(-np.pi, np.pi, 25)
                axes[1, 0].hist(all_yaw, bins=bins, alpha=0.55, color="#808d97", label="All")
                axes[1, 0].hist(success_yaw, bins=bins, alpha=0.55, color="#1d875a", label="Success")
                axes[1, 0].set(xlabel="Yaw (rad)", ylabel="Count", title="Initial orientation")
                axes[1, 0].legend(fontsize=8)
            for axis, key, title in ((axes[1, 1], "attempted_grid", "All attempts"),
                                     (axes[1, 2], "successful_grid", "Successful demos")):
                values = np.asarray(summary[key]).T
                axis.imshow(values, origin="lower", cmap="YlGn", vmin=0,
                            vmax=max(np.max(summary["attempted_grid"]), 1))
                for y, x in np.ndindex(values.shape):
                    axis.text(x, y, str(values[y, x]), ha="center", va="center", fontsize=8)
                axis.set(xlabel="Coarse x bin", ylabel="Coarse y bin", title=title)
            figure.suptitle(f"{split}: {name} | all native-reset attempts vs successful demonstrations")
            figure.tight_layout()
            figure.savefig(destination / f"coverage-{split}-{name}.png", dpi=145)
            plt.close(figure)


def _position_maps(config, details, banks, destination):
    original = banks["official-range100"]["bank"]["config"]["official_center_bounds"][CHEESE]
    all_entries = [entry for info in banks.values() for entry in info["bank"]["states"]]
    limits = {}
    for space in ("world", "relative-bowl"):
        points = np.asarray([_map_xy(entry, space) for entry in all_entries])
        limits[space] = _limits(points)
    for group, info in banks.items():
        entries = info["bank"]["states"]
        count = group_count(config, group)
        for space in ("world", "relative-bowl"):
            positions = np.asarray([_map_xy(entry, space) for entry in entries])
            figure, axes = plt.subplots(2, 5, figsize=(21, 9), sharex=True, sharey=True)
            for row_index, modality in enumerate(config["modalities"]):
                for col_index, step in enumerate(config["checkpoint_steps"]):
                    axis = axes[row_index, col_index]
                    outcomes = np.asarray(details[f"{modality}-{step}-{group}"]["success_by_initial_state"],
                                          dtype=bool)
                    axis.scatter(positions[outcomes, 0], positions[outcomes, 1], color="#198759", s=16,
                                 label="Success")
                    axis.scatter(positions[~outcomes, 0], positions[~outcomes, 1], color="#c83742", s=20,
                                 marker="x", label="Timeout")
                    axis.set(xlim=limits[space][0], ylim=limits[space][1], aspect="equal",
                             title=f"{modality.upper()} {step // 1000}k: {sum(outcomes)}/{count}")
                    if space == "world":
                        minimum, maximum = original["minimum_xy_m"], original["maximum_xy_m"]
                        axis.add_patch(Rectangle(minimum, maximum[0] - minimum[0], maximum[1] - minimum[1],
                                                 fill=False, linestyle="--", edgecolor="#505c75"))
                    axis.grid(alpha=0.15)
                    if row_index == 1:
                        axis.set_xlabel("Cheese x (m)" if space == "world" else "Cheese - bowl x (m)")
                    if col_index == 0:
                        axis.set_ylabel("Cheese y (m)" if space == "world" else "Cheese - bowl y (m)")
            axes[0, 0].legend(fontsize=8)
            figure.suptitle(GROUP_TITLES[group] + f" | {space} | matched scale across both distributions")
            figure.tight_layout()
            figure.savefig(destination / f"{group}-{space}-position-maps.png", dpi=145)
            plt.close(figure)


def _map_xy(entry, space):
    cheese = np.asarray(entry["initial_positions"][CHEESE][:2])
    if space == "world":
        return cheese
    return cheese - np.asarray(entry["initial_positions"][BOWL][:2])


def make_figures(config, rows, details, banks, collection, attempts, root):
    destination = ROOT / "reports/figures" / config["label"]
    destination.mkdir(parents=True, exist_ok=True)
    lookup = {(row["distribution"], row["modality"], row["training_step"]): row for row in rows}
    groups = list(config["groups"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, group in zip(axes, groups, strict=True):
        for modality in config["modalities"]:
            rates = [100 * lookup[group, modality, step]["success_rate"] for step in config["checkpoint_steps"]]
            axis.plot(np.asarray(config["checkpoint_steps"]) / 1000, rates, "o-", label=modality.upper(),
                      color=COLORS[modality], linewidth=2)
            for step, rate in zip(config["checkpoint_steps"], rates, strict=True):
                axis.annotate(f"{rate:g}%", (step / 1000, rate),
                              xytext=(0, 9 if modality == "depth" else -16), textcoords="offset points",
                              ha="center", fontsize=9, color=COLORS[modality])
        axis.set(title=GROUP_TITLES[group], xlabel="Training step (thousands)", ylim=(-5, 110),
                 xticks=np.asarray(config["checkpoint_steps"]) / 1000)
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Success rate (%)")
    figure.suptitle("300 independent successful training demos | from scratch | state + language | seed 0")
    figure.tight_layout()
    figure.savefig(destination / "success-curves.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 3.3))
    axis.axis("off")
    cells = [[f"{step:,}"] + [f"{lookup[group, modality, step]['successes']}/{group_count(config, group)} "
              f"({lookup[group, modality, step]['success_rate']:.1%})"
              for group in groups for modality in ("rgb", "depth")]
             for step in config["checkpoint_steps"]]
    table = axis.table(cellText=cells, colLabels=["Training step", "Original RGB", "Original Depth",
                       "3x RGB", "3x Depth"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    axis.set_title("Paired closed-loop success | 100 original + 200 expanded initial states", pad=15)
    figure.tight_layout()
    figure.savefig(destination / "success-table.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, modality in zip(axes, config["modalities"], strict=True):
        metrics = training_metrics(root / "training" / modality, config["training_steps"])
        validation = [item for item in metrics if "validation_loss" in item]
        train_x = [item["step"] / 1000 for item in metrics]
        train_y = np.asarray([item["train_loss"] for item in metrics])
        window = min(40, len(train_y))
        rolling = np.convolve(train_y, np.ones(window) / window, mode="valid")
        axis.plot(train_x[window - 1:], rolling, label=f"Train (rolling {window} logs)", color=COLORS[modality])
        axis.plot([item["step"] / 1000 for item in validation],
                  [item["validation_loss"] for item in validation], label="Validation", color="#278451")
        axis.set(xlabel="Optimizer step (thousands)", ylabel="Masked action loss", title=modality.upper())
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("Training and independent validation action loss (not closed-loop success)")
    figure.tight_layout()
    figure.savefig(destination / "loss-curves.png", dpi=170)
    plt.close(figure)
    make_coverage_figures(config, attempts, collection["coverage"], destination)
    _position_maps(config, details, banks, destination)
    return destination


def write_results(config, rows, details, pairs, figures, training, collection, banks):
    destination = ROOT / "reports" / config["label"]
    total = sum(row["attempts"] for row in rows)
    if total != 3000 or len(rows) != 20:
        raise ValueError("Not all 3000 formal closed-loop trials were audited")
    with (destination / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage = evaluation_coverage(banks)
    atomic_json(destination / "results.json", {
        "completed_at": datetime.now(timezone.utc).isoformat(), "protocol": config,
        "collection_audit": str(ROOT / "reports" / f"{config['label']}-collection-audit.json"),
        "training_audit": training, "total_attempts": total,
        "attempts_definition": "Completed policy trials, either first official success or full-budget timeout",
        "rows": rows, "paired_counts": pairs, "evaluation_banks": {
            group: {"path": str(info["path"]), "sha256": info["sha256"]}
            for group, info in banks.items()}, "evaluation_coverage": coverage,
        "figures": str(figures), "complete": True})
    lookup = {(row["distribution"], row["modality"], row["training_step"]): row for row in rows}
    lines = ["# 奶酪盒：300条独立初态示范、RGB / Depth 从零训练50k", "",
             f"BDDL官方指令为 {config['official_instruction']}；LIBERO suite根据任务名派生的旧语言为 "
             "Put the cream cheese in the bowl。旧语言试跑不属于本实验，不能混入训练/验证/评估。",
             "成功判据保持官方 On(cream_cheese_1, akita_black_bowl_1)。",
             "两模型均输入双视角所选视觉模态 + 8维机器人状态 + 语言，随机初始化，seed=0，训练50000 optimizer step。",
             "完整RGB-D教师采集300条不同训练初态成功示范及7条独立验证示范；失败/异常均在账本保留。",
             "官方原范围100个新随机初态，奶酪盒中心x/y有效总宽各扩大3倍范围200个独立初态；两模态和全部阶段共用冻结bank。", "",
             "| 训练step | 原范围RGB | 原范围Depth | 3倍范围RGB | 3倍范围Depth |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for step in config["checkpoint_steps"]:
        cells = [f"{lookup[group, modality, step]['successes']}/{group_count(config, group)} "
                 f"({lookup[group, modality, step]['success_rate']:.1%})"
                 for group in config["groups"] for modality in ("rgb", "depth")]
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    lines += ["", "## 教师采集", ""]
    for split in ("train", "validation"):
        stats = collection["splits"][split]
        lines.append(f"- {split}: {stats['successes']}/{stats['attempts']} "
                     f"({stats['success_rate']:.1%})，各失败类型 {stats['termination_counts']}。")
    lines.append(f"- 异常采集恢复记录（仅诊断后使用新seed）：{collection['exception_recoveries']}。")
    lines += ["", "## 位置范围（米）", "",
              "原始有效中心的候选范围为x[-0.07,-0.03]、y[0.11,0.15]；扩大后为x[-0.11,0.01]、y[0.07,0.19]。",
              "原配置的uniform端点因奶酪盒水平半径扣减而反向；这些是碰撞拒绝前的候选支持集，不代表最终均匀采样。"]
    for group, evidence in coverage.items():
        cheese = evidence["sampled_object_ranges"][CHEESE]
        lines.append(f"- {group}: 奶酪盒世界x/y实测 "
                     f"{cheese['minimum_xyz_m'][:2]} 至 {cheese['maximum_xyz_m'][:2]}；"
                     f"相对碗x/y实测 {evidence['cheese_minus_bowl_minimum_xy_m']} 至 "
                     f"{evidence['cheese_minus_bowl_maximum_xy_m']}。")
    collection_audit_path = ROOT / "reports" / f"{config['label']}-collection-audit.json"
    training_audit_path = ROOT / "reports" / f"{config['label']}-training-audit.json"
    lines += ["- 全部尝试及成功子集的各物体x/y/朝向、奶酪盒相对碗位置与6×6覆盖格详见collection-audit.json和coverage-*.png。",
              "", "## 协议与审计", "",
              "- 20Hz、10步稳定、最多300个任务控制步；学生预测8步，每4步重规划；逐步检测官方On谓词。",
              "- 非成功的正式评估结果均为完整预算timeout；异常与中断必须单独诊断，不计入成功率。",
              "- 官方范围是原生随机reset，不是官方固定50初态重复；扩展仅变奶酪盒中心x/y，碗和其他物体保持原始随机化。",
              "- 每条评估轨迹保存全动作/状态/时间和逐帧RGB-D哈希；首动作块由对应固定权重重算，20组固定ID 0精确回放及哈希验证。",
              "- 训练/验证/测试初态及种子互不重叠；训练统计与语言词表仅由300条成功训练示范生成。",
              f"- 采集源码指纹有 {len(collection['collector_source_audit']['versions'])} 种；回放核验器之外的采集实现哈希一致。",
              "- 世界坐标位置图和奶酪盒减碗的相对位置图分别采用跨两分布一致的坐标尺度。",
              "- 单任务、单训练seed；动作loss不是闭环成功率，不从单任务推断模态普遍优劣。", "",
              f"采集审计：{collection_audit_path}",
              f"训练审计：{training_audit_path}",
              f"权重/评估轨迹：{ROOT / 'runs' / config['label']}",
              f"图像目录：{figures}",
              f"位置范围推导：{ROOT / 'docs/cream-cheese-placement.md'}",
              f"采集期间审计器源码变更：{ROOT / 'reports/cream-cheese-on-source-provenance.md'}",
              "复现命令：bash scripts/sim_cpu.sh -u -m scripts.cream_cheese50k_report", ""]
    (destination / "results.md").write_text("\n".join(lines))
    return destination / "results.md"


def main():
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--verify-data-only", action="store_true")
    flags.add_argument("--verify-training-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/cream_cheese50k_on_experiment.json").read_text())
    if (config["training_limit"] != 300 or config["training_steps"] != 50000
            or config["official_instruction"] != "Put the cream cheese on the bowl"
            or config["checkpoint_steps"] != [10000, 20000, 30000, 40000, 50000]
            or set(config["modalities"]) != {"depth", "rgb"}
            or {group: settings["count"] for group, settings in config["groups"].items()}
            != {"official-range100": 100, "expanded3x200": 200}):
        raise ValueError("Cream-cheese protocol must define 300 demonstrations and 3000 rollouts")
    root = ROOT / "runs" / config["label"]
    collection, selected, by_id = audit_collection(config)
    if args.verify_data_only:
        audit_teacher_replays(config, selected, collection)
        print(json.dumps({"collection_verified": True, "attempts": collection["recorded_attempts"],
                          "train": 300, "validation": 7}), flush=True)
        return
    training = verify_training(config, root, collection)
    if args.verify_training_only:
        print("Training fingerprints and all ten fixed checkpoints verified", flush=True)
        return
    banks = validate_evaluation_banks(config, collection)
    rows, details, pairs = audit_evaluations(config, root, training, banks)
    figures = make_figures(config, rows, details, banks, collection, by_id, root)
    report = write_results(config, rows, details, pairs, figures, training, collection, banks)
    print(json.dumps({"complete": True, "attempts": 3000, "report": str(report)}), flush=True)


if __name__ == "__main__":
    main()
