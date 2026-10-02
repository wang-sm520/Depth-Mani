"""Continue the recorded paper-bag experiment, audit it and export its best model."""

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


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-stages", action="store_true",
                        help="Continue after a completed stage; never overwrite a partial training run")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    parent = ROOT / config["parent_run"]
    parent_run = json.loads((parent / "run.json").read_text())
    if parent_run["status"] != "complete" or parent_run["completed_steps"] != config["start_step"]:
        raise ValueError("The parent run has not completed the requested starting step")
    if config["target_step"] <= config["start_step"]:
        raise ValueError("The continuation target must be after the parent step")
    state_path = ROOT / config["state"]
    signature = sha256_file(config_path)
    if state_path.exists():
        if not args.resume_stages:
            raise FileExistsError("Continuation already exists; inspect state before --resume-stages")
        state = json.loads(state_path.read_text())
        if state["config_sha256"] != signature:
            raise ValueError("Continuation configuration changed")
    else:
        if args.resume_stages:
            raise FileNotFoundError("There is no previous continuation state")
        state = {"experiment_id": config["experiment_id"], "config_sha256": signature,
                 "started_at": now(), "completed_stages": [], "commands": [],
                 "start_step": config["start_step"], "target_step": config["target_step"],
                 "robot_executed": False}
    logs = ROOT / config["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt(f"Continuation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    python = sys.executable
    training = [python, "-u", "scripts/resume_airbot_depth.py",
                "--parent-run", str(parent), "--parent-audit", str(ROOT / config["parent_audit"]),
                "--output", str(ROOT / config["run"]), "--steps", str(config["target_step"]),
                "--device", config["device"], "--threads", str(config["threads"])]
    audit = [python, "-u", "scripts/audit_airbot_resume.py",
             "--run", str(ROOT / config["run"]), "--output", str(ROOT / config["report"]),
             "--device", config["device"], "--threads", str(config["threads"])]
    export = [python, "-u", "scripts/export_airbot_depth.py",
              "--checkpoint", str(ROOT / config["run"] / "best.pt"),
              "--audit", str(ROOT / config["report"] / "report.json"),
              "--output", str(ROOT / config["bundle"]), "--device", config["device"],
              "--threads", str(config["threads"])]
    try:
        for stage, command in [("train", training), ("audit", audit), ("export", export)]:
            if stage in state["completed_stages"]:
                continue
            state.update(status="running", stage=stage, updated_at=now())
            atomic_json(state_path, state)
            log_path = logs / f"{stage}-{now().replace(':', '-')}.log"
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid, log=str(log_path))
                atomic_json(state_path, state)
                returncode = child.wait()
            state["commands"].append({"stage": stage, "command": command,
                                       "returncode": returncode, "log": str(log_path), "finished_at": now()})
            if returncode:
                raise RuntimeError(f"{stage} failed with exit {returncode}; inspect {log_path}")
            state["completed_stages"].append(stage)
            atomic_json(state_path, state)
        state.update(status="complete", stage="complete", child_pid=None, completed_at=now())
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
                      "bundle": str(ROOT / config["bundle"]), "robot_executed": False}), flush=True)


if __name__ == "__main__":
    main()
