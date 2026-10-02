import collections
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

from depth_policy.common import ROOT, atomic_json, sha256_file


PROTOCOL = ROOT / "configs/wine50k_experiment.json"


def main():
    config = json.loads(PROTOCOL.read_text())
    label = config["label"]
    root = ROOT / "runs" / label
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(config, root)


def run(config, root):
    label = config["label"]
    state_path = ROOT / "runtime" / f"{label}-state.json"
    logs = ROOT / "runtime/logs" / label
    logs.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(PROTOCOL)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "protocol_sha256": digest, "completed_stages": [], "commands": [], "started_at": now()}
    if state["protocol_sha256"] != digest:
        raise ValueError("Experiment protocol changed; choose a new experiment identity")
    if "error" in state:
        state.setdefault("prior_errors", []).append({"error": state.pop("error"),
                                                     "recorded_at": state["updated_at"]})
        state["active_evaluations"] = {}
    atomic_json(root / "protocol.json", config)

    def update(stage):
        state["stage"] = stage
        state["updated_at"] = now()
        atomic_json(state_path, state)
        print(json.dumps({"stage": stage, "updated_at": state["updated_at"]}), flush=True)

    def command(stage, arguments, cpu_set, gpu=False):
        if stage in state["completed_stages"]:
            return
        update(stage)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0" if gpu else ""
        environment["OMP_NUM_THREADS"] = "4"
        environment["OPENBLAS_NUM_THREADS"] = "4"
        arguments = ["taskset", "-c", cpu_set, "nice", "-n", "10", *map(str, arguments)]
        log_path = logs / f"{stage}.log"
        with log_path.open("a") as log:
            result = subprocess.run(arguments, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state["commands"].append({"stage": stage, "arguments": arguments, "returncode": result.returncode,
                                   "log": str(log_path), "finished_at": now()})
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see {log_path}; no automatic retry")
        state["completed_stages"].append(stage)
        update(stage + "-completed")

    try:
        if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
            raise RuntimeError("Require at least 10 GiB free disk space")
        banks = {}
        for group, settings in config["groups"].items():
            path = ROOT / "data/wine/evaluation" / label / f"{group}.json"
            if not path.exists():
                command(f"prepare-{group}", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                        "depth_policy.evaluate_suite", "prepare", "--config", settings["config"],
                        "--mode", "random", "--count", config["episodes_per_group"],
                        "--seed", settings["seed"], "--bank", path], "0-3")
            banks[group] = path
        seen = set()
        prior_hashes = set()
        for prior in (ROOT / "data/wine/evaluation").glob("*.json"):
            for entry in json.loads(prior.read_text()).get("states", []):
                prior_hashes.add(entry["sha256"])
        for group, path in banks.items():
            bank = json.loads(path.read_text())
            settings = config["groups"][group]
            assert bank["seed"] == settings["seed"] and bank["mode"] == "random"
            assert bank["config"] == json.loads((ROOT / settings["config"]).read_text())
            assert len(bank["states"]) == config["episodes_per_group"]
            hashes = {entry["sha256"] for entry in bank["states"]}
            assert len(hashes) == config["episodes_per_group"] and not hashes & (seen | prior_hashes)
            assert all(entry["reset_and_settle_reproduced"] for entry in bank["states"])
            seen.update(hashes)
        state["banks"] = {group: {"path": str(path), "sha256": sha256_file(path)} for group, path in banks.items()}
        update("banks-frozen")
        for modality in config["modalities"]:
            output = root / "training" / modality
            arguments = [config["training_python"], "-u", "-m", "depth_policy.train",
                         "--data", config["training_data"], "--output", output,
                         "--modality", modality, "--device", "cuda", "--steps", config["training_steps"],
                         "--limit", config["training_limit"], "--seed", config["training_seed"],
                         "--save-every", 500, "--checkpoint-every", 10000]
            if (output / "latest.pt").exists() and f"train-{modality}" not in state["completed_stages"]:
                raise RuntimeError("Interrupted training requires explicit checkpoint diagnosis before resuming")
            command(f"train-{modality}", arguments, config["training_cpu_set"], gpu=True)
        command("verify-training", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                "scripts.wine50k_report", "--verify-training-only"], "8-11")
        jobs = collections.deque()
        for step in config["checkpoint_steps"]:
            for modality in config["modalities"]:
                for group, bank_path in banks.items():
                    stage = f"eval-{modality}-{step}-{group}"
                    if stage not in state["completed_stages"]:
                        output = root / "evaluation" / f"{modality}-{step}" / group
                        checkpoint = root / "training" / modality / f"step_{step:06d}.pt"
                        jobs.append((stage, ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                                     "depth_policy.evaluate_suite", "run", "--bank", bank_path,
                                     "--checkpoint", checkpoint, "--compact", "--replan-steps",
                                     config["replan_steps"], "--output", output]))
        active = {}
        available = collections.deque(config["evaluation_cpu_sets"])
        update("evaluation-running")
        try:
            while jobs or active:
                while jobs and available:
                    stage, arguments = jobs.popleft()
                    cpu_set = available.popleft()
                    arguments = ["taskset", "-c", cpu_set, "nice", "-n", "10", *map(str, arguments)]
                    log_path = logs / f"{stage}.log"
                    log = log_path.open("a")
                    process = subprocess.Popen(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    active[stage] = (process, cpu_set, log, arguments, log_path)
                state["active_evaluations"] = {name: value[0].pid for name, value in active.items()}
                state["queued_evaluations"] = len(jobs)
                state["updated_at"] = now()
                atomic_json(state_path, state)
                time.sleep(10)
                for stage, (process, cpu_set, log, arguments, log_path) in list(active.items()):
                    if process.poll() is None:
                        continue
                    log.close()
                    del active[stage]
                    available.append(cpu_set)
                    state["commands"].append({"stage": stage, "arguments": arguments,
                                             "returncode": process.returncode, "log": str(log_path),
                                             "finished_at": now()})
                    if process.returncode:
                        raise RuntimeError(f"{stage} failed; see {log_path}")
                    state["completed_stages"].append(stage)
                    update("evaluation-running")
        finally:
            for process, cpu_set, log, arguments, log_path in active.values():
                process.terminate()
            for process, cpu_set, log, arguments, log_path in active.values():
                process.wait()
                log.close()
        command("audit-and-report", ["bash", "scripts/sim_cpu.sh", "-u", "-m",
                "scripts.wine50k_report"], "8-11")
        state["complete"] = True
        state["active_evaluations"] = {}
        update("completed")
    except BaseException as error:
        state["error"] = repr(error)
        update("failed-needs-diagnosis")
        raise


def now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
