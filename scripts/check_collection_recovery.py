import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from depth_policy.common import atomic_json, sha256_file
    from depth_policy.episodes import episode_index

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/smoke-recovery")
    parser.add_argument("--report", default="reports/collection-recovery.json")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Recovery test needs a fresh isolated output directory")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    command = ["bash", "scripts/sim_cpu.sh", "-u", "-m", "depth_policy.collect", "--dummy",
               "--output", str(destination), "--attempts", "2", "--max-steps", "300"]
    log_path = Path(args.report).with_suffix(".log")
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Collector exited before interruption; see {log_path}")
                partials = list(destination.glob("*.partial.h5"))
                if partials and partials[0].stat().st_size >= 6_000_000:
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError("Collector did not produce a moving trajectory within 60 seconds")
            os.kill(process.pid, signal.SIGTERM)
            if process.wait(timeout=30) != 0:
                raise RuntimeError("Graceful termination returned an error")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
    interrupted = episode_index(destination)
    if len(interrupted) != 1 or interrupted[0]["termination_reason"] != "interrupted":
        raise AssertionError("Interrupted run must preserve exactly one interrupted episode")
    if list(destination.glob("*.partial.h5")):
        raise AssertionError("Graceful interruption left partial data")
    original_hash = sha256_file(interrupted[0]["path"])
    with log_path.open("a") as log:
        subprocess.run([*command[:-4], "--attempts", "1", "--max-steps", "5"], cwd=root, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    records = episode_index(destination)
    if len(records) != 2 or len({record["episode_id"] for record in records}) != 2:
        raise AssertionError("Resume did not create a unique second episode")
    if sha256_file(interrupted[0]["path"]) != original_hash:
        raise AssertionError("Resume modified the original episode")
    if list(destination.glob("*.partial.h5")):
        raise AssertionError("Resume left incomplete output")
    with log_path.open("a") as log:
        subprocess.run(["bash", "scripts/sim_cpu.sh", "-m", "depth_policy.replay", interrupted[0]["path"],
                        "--report", str(Path(args.report).with_suffix(".replay.json"))], cwd=root,
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        subprocess.run([str(root / ".venv/bin/python"), "-m", "depth_policy.validate", "--data", str(destination),
                        "--report", str(Path(args.report).with_suffix(".validation.json"))], cwd=root,
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    report = {"passed": True, "signal": "SIGTERM", "output": str(destination),
              "original_episode_sha256": original_hash, "original_unchanged_after_resume": True,
              "episodes": [{key: record[key] for key in ["path", "episode_id", "initial_state_id", "termination_reason"]}
                           for record in records], "teacher_data_used": False,
              "scope": "Graceful termination and resume of a real simulated scripted-motion collector, not a teacher success test"}
    atomic_json(args.report, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
