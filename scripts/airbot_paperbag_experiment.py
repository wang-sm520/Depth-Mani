"""Run the authorized offline paper-bag conversion, training and audit once."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.lerobot import LeRobotSource


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    state_path = ROOT / config["state"]
    source = LeRobotSource(config["source"], config["camera_keys"])
    if (source.info["total_episodes"] != config["source_episodes"]
            or source.info["total_frames"] != config["source_frames"]
            or source.fps != config["fps"] or source.prompt != config["prompt"]):
        raise ValueError("Actual source dataset differs from the chosen experiment")
    signature = sha256_file(config_path)
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("Experiment already exists; inspect state before explicit --resume")
        state = json.loads(state_path.read_text())
        if state["config_sha256"] != signature:
            raise ValueError("Cannot resume with a different experiment config")
    else:
        if args.resume:
            raise FileNotFoundError("No existing experiment to resume")
        state = {"experiment_id": config["experiment_id"], "config_sha256": signature,
                 "started_at": now(), "status": "running", "completed_stages": [], "commands": [],
                 "robot_executed": False}
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt(f"Experiment interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    python = sys.executable
    conversion = [python, "-u", "-m", "airbot_depth.convert", "--source", config["source"],
                  "--output", str(ROOT / config["converted_data"]), "--device", config["device"],
                  "--cameras", *config["camera_keys"],
                  "--validation-fraction", str(config["validation_fraction"]),
                  "--split-seed", str(config["split_seed"]), "--threads", str(config["threads"]),
                  "--local-files-only"]
    if args.resume and (ROOT / config["converted_data"] / "manifest.json").exists():
        conversion.append("--resume")
    training = [python, "-u", "-m", "airbot_depth.train", "--data", str(ROOT / config["converted_data"]),
                "--output", str(ROOT / config["run"]), "--steps", str(config["train_steps"]),
                "--device", config["device"], "--batch-size", str(config["batch_size"]),
                "--horizon", str(config["horizon"]), "--seed", str(config["train_seed"]),
                "--learning-rate", str(config["learning_rate"]), "--threads", str(config["threads"]),
                "--checkpoint-every", str(config["checkpoint_every"]),
                "--eval-every", str(config["eval_every"])]
    audit = [python, "-u", "-m", "airbot_depth.audit", "--data", str(ROOT / config["converted_data"]),
             "--run", str(ROOT / config["run"]), "--output", str(ROOT / config["report"]),
             "--device", config["device"]]
    try:
        for stage, command in [("convert", conversion), ("train", training), ("audit", audit)]:
            if stage in state["completed_stages"]:
                continue
            state.update(status="running", stage=stage, updated_at=now())
            atomic_json(state_path, state)
            log_path = logs / f"{stage}-{now().replace(':', '-')}.log"
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                state["log"] = str(log_path)
                atomic_json(state_path, state)
                returncode = child.wait()
            state["commands"].append({"stage": stage, "command": command, "returncode": returncode,
                                       "log": str(log_path), "finished_at": now()})
            if returncode:
                raise RuntimeError(f"{stage} failed with exit {returncode}; inspect {log_path}")
            state["completed_stages"].append(stage)
            atomic_json(state_path, state)
        state.update(status="complete", stage="complete", completed_at=now(), child_pid=None)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}", updated_at=now())
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                state["child_still_stopping"] = True
        atomic_json(state_path, state)
        raise
    atomic_json(state_path, state)
    print(json.dumps({"status": "complete", "report": str(ROOT / config["report"]),
                      "robot_executed": False}), flush=True)


if __name__ == "__main__":
    main()
