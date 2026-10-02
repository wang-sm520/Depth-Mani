import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from urllib.request import urlopen

from depth_policy.common import ROOT, atomic_json, sha256_file


PROTOCOL = ROOT / "configs/cream_cheese50k_on_experiment.json"
COLLECT_CONFIG = "configs/cream_cheese_on.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_banks(config):
    seen_seeds = set()
    seen_hashes = set()
    results = {}
    attempt_bank = ROOT / config["attempt_bank"]
    attempts = json.loads(attempt_bank.read_text())
    assert attempts["schema_version"] == 1
    for split, count in (("train", config["train_attempts"]),
                         ("validation", config["validation_attempts"])):
        candidates = [entry for entry in attempts["states"] if entry["split"] == split]
        assert len(candidates) == count
        seeds = {entry["seed"] for entry in candidates}
        assert len(seeds) == count and not seeds & seen_seeds
        seen_seeds.update(seeds)
    ledger = json.loads((ROOT / config["training_data"] / "attempt-index.json").read_text())
    assert ledger["bank_sha256"] == sha256_file(attempt_bank)
    for entry in ledger["entries"]:
        digest = entry.get("initial_state_sha256")
        if digest:
            assert digest not in seen_hashes
            seen_hashes.add(digest)
    for group, settings in config["groups"].items():
        path = ROOT / "data/cream_cheese/evaluation" / config["label"] / f"{group}.json"
        bank = json.loads(path.read_text())
        assert bank["mode"] == "random" and bank["seed"] == settings["seed"]
        assert bank["config"] == json.loads((ROOT / settings["config"]).read_text())
        assert len(bank["states"]) == settings["count"]
        assert all(entry["reset_and_settle_reproduced"] for entry in bank["states"])
        hashes = {entry["sha256"] for entry in bank["states"]}
        assert len(hashes) == settings["count"] and not hashes & seen_hashes
        seeds = {entry["seed"] for entry in bank["states"]}
        assert len(seeds) == settings["count"] and not seeds & seen_seeds
        seen_seeds.update(seeds)
        seen_hashes.update(hashes)
        results[group] = {"path": str(path), "sha256": sha256_file(path), "count": settings["count"]}
    return results


def run(config, root, pilot_only=False):
    label = config["label"]
    state_path = ROOT / "runtime" / f"{label}-state.json"
    logs = ROOT / "runtime/logs" / label
    logs.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(PROTOCOL)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "protocol_sha256": digest, "completed_stages": [], "commands": [], "started_at": now()}
    if state["protocol_sha256"] != digest:
        raise ValueError("Experiment protocol changed; choose a new experiment identity")
    if state.get("error"):
        raise RuntimeError("Previous stage failed; diagnose preserved files before restarting")
    if state.get("active_evaluations"):
        raise RuntimeError("Evaluation processes were active when the coordinator stopped; diagnose their outputs")
    if state.get("teacher_server"):
        raise RuntimeError("Teacher server ownership was not cleared; diagnose the recorded PID and service")
    if not pilot_only and not {"pilot-train", "pilot-validation"} <= set(state["completed_stages"]):
        raise RuntimeError("Finish and review the frozen-seed teacher pilot before full collection")
    atomic_json(root / "protocol.json", config)

    def update(stage):
        state["stage"] = stage
        state["updated_at"] = now()
        atomic_json(state_path, state)
        print(json.dumps({"stage": stage, "updated_at": state["updated_at"]}), flush=True)

    def command(stage, args, cpu_set, gpu=False):
        if stage in state["completed_stages"]:
            return
        if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
            raise RuntimeError("Require at least 10 GiB free disk space")
        update(stage)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0" if gpu else ""
        environment["OMP_NUM_THREADS"] = "4"
        environment["OPENBLAS_NUM_THREADS"] = "4"
        args = ["taskset", "-c", cpu_set, "nice", "-n", "10", *map(str, args)]
        log_path = logs / f"{stage}.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=220)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        state["commands"].append({"stage": stage, "arguments": args, "returncode": returncode,
                                  "log": str(log_path), "finished_at": now()})
        if returncode:
            raise RuntimeError(f"{stage} failed; see {log_path}; no automatic retry")
        state["completed_stages"].append(stage)
        update(stage + "-completed")

    def collector(stage, split, attempts):
        bank = ROOT / config["attempt_bank"]
        command(stage, ["bash", "scripts/sim_cpu.sh", "-u", "-m", "depth_policy.collect_random",
                        "run", "--config", COLLECT_CONFIG, "--bank", bank, "--split", split,
                        "--attempts", attempts, "--output", ROOT / config["training_data"]], "4-7")

    def collect_with_teacher(pilot):
        stages = [("pilot-train", "train", config["pilot_train_attempts"]),
                  ("pilot-validation", "validation", config["pilot_validation_attempts"])]
        if not pilot:
            stages += [("collect-train", "train", config["train_attempts"]),
                       ("collect-validation", "validation", config["validation_attempts"])]
        if all(stage in state["completed_stages"] for stage, _, _ in stages):
            return
        teacher_config = json.loads((ROOT / COLLECT_CONFIG).read_text())
        port = teacher_config["teacher_port"]
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Teacher port {port} is in use; do not take over another server")
        openpi = Path(teacher_config["openpi_root"])
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"})
        server_log = logs / f"teacher-{now().replace(':', '-')}.log"
        update("starting-teacher")
        with server_log.open("w") as log:
            server = subprocess.Popen([
                "taskset", "-c", "0-3", "nice", "-n", "10", str(openpi / ".venv/bin/python"),
                "-u", "scripts/serve_teacher.py", "--platform", "cpu", "--port", str(port),
                "--openpi-root", str(openpi), "--checkpoint", teacher_config["teacher_checkpoint"]],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state["teacher_server"] = {"pid": server.pid, "log": str(server_log)}
            update("waiting-for-teacher")
            try:
                for _ in range(180):
                    if server.poll() is not None:
                        raise RuntimeError(f"Teacher exited during startup: {server_log}")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(1)
                else:
                    raise RuntimeError(f"Teacher startup timed out: {server_log}")
                for stage, split, attempts in stages:
                    collector(stage, split, attempts)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                state.pop("teacher_server", None)
                update("teacher-stopped")

    try:
        command("manifest", ["bash", "scripts/sim_cpu.sh", "-u", "-m", "depth_policy.manifest",
                             "--config", COLLECT_CONFIG], "4-7")
        command("prepare-attempts", ["bash", "scripts/sim_cpu.sh", "-u", "-m", "depth_policy.collect_random",
                                     "prepare", "--config", COLLECT_CONFIG,
                                     "--bank", ROOT / config["attempt_bank"],
                                     "--train-attempts", config["train_attempts"],
                                     "--validation-attempts", config["validation_attempts"],
                                     "--train-seed", config["train_seed"],
                                     "--validation-seed", config["validation_seed"]], "4-7")
        collect_with_teacher(pilot_only)
        if pilot_only:
            update("pilot-completed-needs-feasibility-review")
            return
        command("verify-data", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                                "scripts.cream_cheese50k_report", "--verify-data-only"], "8-11")
        for group, settings in config["groups"].items():
            command(f"prepare-{group}", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                                         "depth_policy.evaluate_suite", "prepare", "--config", settings["config"],
                                         "--mode", "random", "--count", settings["count"],
                                         "--seed", settings["seed"], "--bank",
                                         ROOT / "data/cream_cheese/evaluation" / label / f"{group}.json"], "4-7")
        frozen = frozen_banks(config)
        if "banks" in state and state["banks"] != frozen:
            raise ValueError("A frozen evaluation bank changed since the first run")
        state["banks"] = frozen
        update("banks-frozen")
        for modality in config["modalities"]:
            output = root / "training" / modality
            if (output / "latest.pt").exists() and f"train-{modality}" not in state["completed_stages"]:
                raise RuntimeError("Interrupted training needs explicit checkpoint diagnosis")
            command(f"train-{modality}", [config["training_python"], "-u", "-m", "depth_policy.train",
                                            "--data", config["training_data"], "--output", output,
                                            "--modality", modality, "--device", "cuda", "--steps",
                                            config["training_steps"], "--limit", config["training_limit"],
                                            "--seed", config["training_seed"], "--save-every", 500,
                                            "--checkpoint-every", 10000], config["training_cpu_set"], gpu=True)
        command("verify-training", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                                    "scripts.cream_cheese50k_report", "--verify-training-only"], "8-11")
        jobs = collections.deque()
        for step in config["checkpoint_steps"]:
            for modality in config["modalities"]:
                for group, details in state["banks"].items():
                    stage = f"eval-{modality}-{step}-{group}"
                    if stage not in state["completed_stages"]:
                        jobs.append((stage, ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                                             "depth_policy.evaluate_suite", "run", "--bank", details["path"],
                                             "--checkpoint", root / "training" / modality / f"step_{step:06d}.pt",
                                             "--compact", "--replan-steps", config["replan_steps"],
                                             "--output", root / "evaluation" / f"{modality}-{step}" / group]))
        active = {}
        available = collections.deque(config["evaluation_cpu_sets"])
        update("evaluation-running")
        try:
            while jobs or active:
                while jobs and available:
                    stage, args = jobs.popleft()
                    cpu_set = available.popleft()
                    args = ["taskset", "-c", cpu_set, "nice", "-n", "10", *map(str, args)]
                    log_path = logs / f"{stage}.log"
                    log = log_path.open("a")
                    process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    active[stage] = (process, cpu_set, log, args, log_path)
                state["active_evaluations"] = {stage: job[0].pid for stage, job in active.items()}
                state["queued_evaluations"] = len(jobs)
                update("evaluation-running")
                time.sleep(10)
                for stage, (process, cpu_set, log, args, log_path) in list(active.items()):
                    if process.poll() is None:
                        continue
                    log.close()
                    del active[stage]
                    available.append(cpu_set)
                    state["commands"].append({"stage": stage, "arguments": args, "returncode": process.returncode,
                                              "log": str(log_path), "finished_at": now()})
                    if process.returncode:
                        raise RuntimeError(f"{stage} failed; inspect {log_path}; no automatic retry")
                    state["completed_stages"].append(stage)
        finally:
            for process, _, log, _, _ in active.values():
                process.terminate()
            for process, _, log, _, _ in active.values():
                try:
                    process.wait(timeout=220)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            if active:
                state["interrupted_evaluations"] = {stage: str(job[4]) for stage, job in active.items()}
                state["active_evaluations"] = {}
                update("evaluation-interrupted-needs-diagnosis")
        state["active_evaluations"] = {}
        update("evaluation-completed")
        command("audit-and-report", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                                     "scripts.cream_cheese50k_report"], "8-11")
        state["complete"] = True
        state["active_evaluations"] = {}
        update("completed")
    except BaseException as error:
        state["error"] = repr(error)
        update("failed-needs-diagnosis")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    def request_stop(signum, frame):
        raise InterruptedError(f"Coordinator received signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = json.loads(PROTOCOL.read_text())
    root = ROOT / "runs" / config["label"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(config, root, args.pilot_only)


if __name__ == "__main__":
    main()
