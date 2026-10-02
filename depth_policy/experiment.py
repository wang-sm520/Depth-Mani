import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from depth_policy.common import ROOT, atomic_json, load_config
from depth_policy.episodes import episode_index
from depth_policy.data import data_fingerprint, training_episodes
from depth_policy.summarize import summarize


def pilot_gate(report):
    if report["incomplete_attempts"]:
        raise ValueError("Pilot still has incomplete attempts")
    training = report["splits"]["train"]
    if training["attempts"] < 20 or training["unique_initial_states"] < 20:
        raise ValueError("Pilot must cover at least 20 different initial states")
    if training["success_rate"] < 0.8:
        raise ValueError("Teacher pilot below the operational 80% gate; diagnose before bulk collection")
    if any(reason not in {"success", "timeout"} for reason in training["termination_counts"]):
        raise ValueError("Pilot contains infrastructure failures or interrupted attempts")
    if not report["policies"] or any(policy.get("policy_config") != "pi05_libero" for policy in report["policies"]):
        raise ValueError("Pilot is not from the official pi05_libero teacher")


def physical_resources(minimum_disk_gib=20):
    free_disk = shutil.disk_usage(ROOT).free / 2**30
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    available_gib = available_kib / 2**20
    if free_disk < minimum_disk_gib or available_gib < 8:
        raise RuntimeError(f"Resource gate: disk {free_disk:.1f} GiB, available RAM {available_gib:.1f} GiB")
    return {"disk_free_GiB": free_disk, "memory_available_GiB": available_gib}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.data_dir = self.config["data_dir"]
        self.run_root = Path("runs") / self.config["task"]
        self.child = None
        self.state_path = Path(f"runtime/{args.label}-state.json")
        self.state = {"pid": os.getpid(), "label": args.label, "config": vars(args), "status": "starting",
                      "goal_complete": False, "completed_stages": [], "commands": []}
        self.environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                            "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}

    def save(self, stage):
        self.state["stage"] = stage
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.state_path, self.state)
        print(json.dumps({"stage": stage, "status": self.state["status"]}), flush=True)

    def stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        os.killpg(self.child.pid, signal.SIGTERM)
        try:
            self.child.wait(timeout=220)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()

    def interrupt(self, signum, frame):
        self.stop_child()
        self.state["status"] = "interrupted"
        self.save(self.state.get("stage", "unknown"))
        raise SystemExit(128 + signum)

    def command(self, stage, arguments, timeout_seconds):
        resources = physical_resources()
        log_path = Path(f"runtime/logs/{self.args.label}-{stage}.log")
        record = {"stage": stage, "argv": arguments, "log": str(log_path), "resources": resources,
                  "timeout_seconds": timeout_seconds}
        self.state["commands"].append(record)
        self.save(stage)
        start = time.monotonic()
        with log_path.open("a") as stream:
            stream.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {arguments!r} ===\n")
            stream.flush()
            environment = dict(self.environment)
            if stage.startswith("train-") and self.args.training_device == "cuda":
                environment["CUDA_VISIBLE_DEVICES"] = "0"
            self.child = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = self.child.pid
            self.save(stage)
            try:
                return_code = self.child.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self.stop_child()
                raise TimeoutError(f"Stage timed out: {stage}; see {log_path}") from None
        record["seconds"] = time.monotonic() - start
        record["return_code"] = return_code
        self.child = None
        if return_code:
            raise RuntimeError(f"Stage {stage} failed ({return_code}); see {log_path}")
        self.state["completed_stages"].append(stage)
        self.save(stage)

    def wait_for_pilot(self):
        self.save("waiting-for-pilot")
        if self.args.pilot_service:
            self.wait_for_pilot_service()
            return
        pid = int(Path(self.args.pilot_pid_file).read_text())
        deadline = time.monotonic() + 4 * 3600
        while Path(f"/proc/{pid}/cmdline").exists():
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            if not command:
                break
            if b"run_teacher_cpu_batch.sh" not in command or b"teacher-cpu-pilot20" not in command:
                raise RuntimeError("Pilot PID was reused or refers to another command")
            if time.monotonic() > deadline:
                raise TimeoutError("Existing pilot did not finish within four hours; it has not been stopped")
            time.sleep(20)
        report = json.loads(Path(self.args.pilot_report).read_text())
        pilot_gate(report)
        self.state["pilot_gate"] = {"attempts": report["splits"]["train"]["attempts"],
                                    "successes": report["splits"]["train"]["successes"],
                                    "unique_initial_states": report["splits"]["train"]["unique_initial_states"]}
        self.state["completed_stages"].append("teacher-pilot-gate")

    def wait_for_pilot_service(self):
        deadline = time.monotonic() + 4 * 3600
        while True:
            result = subprocess.run(["systemctl", "--user", "show", self.args.pilot_service,
                                     "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus"],
                                    check=True, text=True, capture_output=True)
            properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            if properties.get("LoadState") != "loaded":
                raise RuntimeError("Pilot service is missing; cannot verify completion")
            if (properties.get("SubState") == "exited" or
                    properties.get("ActiveState") not in {"active", "activating", "deactivating"}):
                if properties.get("Result") != "success" or properties.get("ExecMainStatus") != "0":
                    raise RuntimeError(f"Pilot service failed: {properties}")
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Pilot service has not finished; it has not been stopped")
            time.sleep(20)
        report = json.loads(Path(self.args.pilot_report).read_text())
        if Path(report["directory"]).resolve() != Path(self.data_dir).resolve():
            raise ValueError("Pilot report belongs to another dataset")
        pilot_gate(report)
        self.state["pilot_gate"] = report["splits"]["train"]
        self.state["completed_stages"].append("teacher-pilot-gate")

    def wait_for_existing_batch(self):
        if not self.args.wait_batch_pid:
            return
        self.save("waiting-for-existing-batch")
        deadline = time.monotonic() + 4 * 3600
        path = Path(f"/proc/{self.args.wait_batch_pid}/cmdline")
        while path.exists():
            command = path.read_bytes()
            if not command:
                break
            if b"run_teacher_cpu_batch.sh" not in command or self.args.wait_batch_label.encode() not in command:
                raise RuntimeError("Existing batch PID identity changed")
            if time.monotonic() > deadline:
                raise TimeoutError("Existing batch still running; it has not been stopped")
            time.sleep(20)
        report = json.loads(Path(f"reports/{self.args.wait_batch_label}-summary.json").read_text())
        if report["incomplete_attempts"]:
            raise RuntimeError("Existing batch left incomplete trajectories")
        self.state["completed_stages"].append("existing-batch-finished")

    def collect(self):
        self.command("validate-pilot", [".venv/bin/python", "-m", "depth_policy.validate",
                     "--config", self.args.config, "--data", self.data_dir,
                     "--report", f"reports/{self.args.label}-pilot-validation.json"], 600)
        for batch in range((self.args.target + 19) // 20 + 3):
            episodes = episode_index(self.data_dir)
            count = sum(item["success"] and item["split"] == "train" for item in episodes)
            if count >= self.args.target:
                break
            stage = f"train-collection-{batch}"
            self.command(stage, ["bash", "scripts/run_teacher_cpu_batch.sh", "20", "train",
                                 f"{self.args.label}-{stage}", str(self.args.target), self.args.config], 180 * 60)
        episodes = episode_index(self.data_dir)
        if sum(item["success"] and item["split"] == "train" for item in episodes) < self.args.target:
            raise RuntimeError("Training success target not met")
        manifest = json.loads(Path(self.config["manifest"]).read_text())
        for split in ["validation", "test"]:
            expected = {entry["id"] for entry in manifest["states"] if entry["split"] == split}
            records = [item for item in episode_index(self.data_dir) if item["split"] == split]
            seen = {item["initial_state_id"] for item in records}
            if not seen <= expected or len(records) != len(seen):
                raise RuntimeError(f"Unexpected duplicate or foreign {split} attempts; audit before resuming")
            remaining = len(expected - seen)
            if remaining:
                stage = f"{split}-collection"
                self.command(stage, ["bash", "scripts/run_teacher_cpu_batch.sh", str(remaining), split,
                                     f"{self.args.label}-{stage}", "", self.args.config], 180 * 60)
        report = summarize(self.data_dir)
        if report["incomplete_attempts"] or any(
            reason not in {"success", "timeout"}
            for split in report["splits"].values() for reason in split["termination_counts"]
        ):
            raise RuntimeError("Incomplete or infrastructure-failed teacher attempts require diagnosis")
        atomic_json(f"reports/{self.args.label}-teacher-summary.json", report)
        self.command("validate-full-data", [".venv/bin/python", "-m", "depth_policy.validate",
                     "--config", self.args.config, "--data", self.data_dir,
                     "--report", f"reports/{self.args.label}-data-validation.json"], 1200)

    def train_and_evaluate(self):
        self.verify_nested_training_data()
        results = {}
        for modality in ["depth", "rgb", "state"]:
            output = str(self.run_root / f"{modality}-{self.args.target}-seed0")
            arguments = ["taskset", "-c", "8-11", "nice", "-n", "10", self.args.training_python, "-u", "-m",
                         "depth_policy.train", "--data", self.data_dir, "--modality", modality,
                         "--device", self.args.training_device, "--steps",
                         str(self.args.steps), "--limit", str(self.args.target), "--seed", "0", "--output", output]
            if (Path(output) / "latest.pt").exists():
                arguments.append("--resume")
            self.command(f"train-{modality}", arguments, 4 * 3600)
            results[modality] = {"run": output, "config": json.loads((Path(output) / "run.json").read_text())}
            for split in ["validation", "test"]:
                destination = str(self.run_root / f"eval-{modality}-{self.args.target}-seed0-{split}")
                self.command(f"eval-{modality}-{split}", ["taskset", "-c", "8-11", "nice", "-n", "10",
                             "bash", "scripts/sim_cpu.sh", "-u", "-m", "depth_policy.evaluate",
                             "--config", self.args.config,
                             "--checkpoint", f"{output}/best.pt", "--device", "cpu", "--split", split,
                             "--repeats", "1", "--output", destination], 3600)
                summary = summarize(destination)
                if any(reason not in {"success", "timeout"}
                       for split_result in summary["splits"].values()
                       for reason in split_result["termination_counts"]):
                    raise RuntimeError("Evaluation contains infrastructure failures; diagnose before comparison")
                manifest = json.loads(Path(self.config["manifest"]).read_text())
                expected = {entry["id"] for entry in manifest["states"] if entry["split"] == split}
                seen = {item["initial_state_id"] for item in episode_index(destination)}
                if summary["incomplete_attempts"] or seen != expected or summary["completed_attempts"] != len(expected):
                    raise RuntimeError("Incomplete or duplicated evaluation; do not claim completion")
                atomic_json(f"reports/{self.args.label}-{modality}-{split}.json", summary)
                results[modality][split] = summary
        comparison = {"target_successful_demonstrations": self.args.target, "training_steps": self.args.steps,
                      "training_seed": 0, "results": results, "goal_complete": False,
                      "next": "Audit closed-loop outcomes; decide and execute conditional 300/500 learning curves."}
        atomic_json(f"reports/{self.args.label}-comparison.json", comparison)

    def verify_nested_training_data(self):
        for size in (100, 300):
            if size >= self.args.target:
                continue
            previous_path = self.run_root / f"depth-{size}-seed0" / "run.json"
            previous = json.loads(previous_path.read_text())
            current = data_fingerprint(training_episodes(self.data_dir, "train", size))
            if previous["data"]["train"] != current:
                raise ValueError(f"Training subset at {size} differs from the prior learning-curve run")
            validation = data_fingerprint(training_episodes(self.data_dir, "validation"))
            if previous["data"]["validation"] != validation:
                raise ValueError("Learning-curve validation data changed")

    def run(self):
        self.state["status"] = "running"
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)
        try:
            self.wait_for_pilot()
            self.wait_for_existing_batch()
            self.collect()
            self.train_and_evaluate()
            self.state["status"] = "first_stage_finished_requires_audit"
            self.save("awaiting-outcome-audit-and-learning-curve-decision")
        except Exception as exception:
            self.stop_child()
            self.state["status"] = "failed"
            self.state["error"] = repr(exception)
            self.save(self.state.get("stage", "unknown"))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="first100")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--pilot-service")
    parser.add_argument("--training-python", default=".venv/bin/python")
    parser.add_argument("--training-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--target", type=int, choices=[100, 300, 500], default=100)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--pilot-pid-file", default="runtime/teacher-cpu-pilot20.pid")
    parser.add_argument("--pilot-report", default="reports/teacher-cpu-pilot20-summary.json")
    parser.add_argument("--wait-batch-pid", type=int)
    parser.add_argument("--wait-batch-label")
    args = parser.parse_args()
    if bool(args.wait_batch_pid) != bool(args.wait_batch_label):
        raise ValueError("Existing batch PID and label must both be supplied")
    if not args.label.replace("-", "").replace("_", "").isalnum() or args.steps < 1:
        raise ValueError("Invalid run label or training step count")
    os.chdir(ROOT)
    Path("runtime/logs").mkdir(parents=True, exist_ok=True)
    with open("runtime/experiment.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if Path(f"runtime/{args.label}-state.json").exists():
            raise FileExistsError("Choose a new experiment label; existing state will not be overwritten")
        Experiment(args).run()


if __name__ == "__main__":
    main()
