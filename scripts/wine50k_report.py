import argparse
import collections
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import h5py
import matplotlib
import numpy as np
import torch

from depth_policy.common import ROOT, atomic_json, sha256_file
from depth_policy.data import data_fingerprint, training_episodes
from depth_policy.episodes import episode_index
from depth_policy.evaluate import Student
from depth_policy.evaluate_suite import check_records
from depth_policy.validate import validate_episode

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def verify_training(config, root):
    original = json.loads((ROOT / "runs/put_the_wine_bottle_on_top_of_the_cabinet/depth-300-seed0/run.json").read_text())
    fingerprint = {split: data_fingerprint(training_episodes(ROOT / config["training_data"], split,
                   config["training_limit"] if split == "train" else None)) for split in ("train", "validation")}
    assert fingerprint == original["data"]
    reports = {}
    for modality in config["modalities"]:
        run = json.loads((root / "training" / modality / "run.json").read_text())
        assert run["initialization"] == "all_student_parameters_random"
        assert not run["config"]["resume"] and run["config"]["seed"] == config["training_seed"]
        assert run["config"]["steps"] == 50000 and run["config"]["checkpoint_every"] == 10000
        assert run["train_episodes"] == 300 and run["train_initial_states"] == 35
        for field in ("data", "statistics", "vocabulary", "train_frames"):
            assert run[field] == original[field]
        reports[modality] = {}
        for step in config["checkpoint_steps"]:
            path = root / "training" / modality / f"step_{step:06d}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["step"] == step and checkpoint["config"]["modality"] == modality
            for field in ("data", "statistics", "vocabulary"):
                assert checkpoint[field] == run[field]
            reports[modality][str(step)] = {"path": str(path), "sha256": sha256_file(path), "step": step}
        latest = torch.load(root / "training" / modality / "latest.pt", map_location="cpu", weights_only=False)
        assert latest["step"] == 50000
    result = {"training_data_unchanged": True, "random_initialization": True,
              "train_episodes": 300, "train_initial_states": 35,
              "train_frames": original["train_frames"], "checkpoints": reports}
    atomic_json(ROOT / "reports" / f"{config['label']}-training-audit.json", result)
    return result


def audit_evaluations(config, root, training):
    rows = []
    details = {}
    pairs = {}
    directory = ROOT / "reports" / config["label"]
    directory.mkdir(parents=True, exist_ok=True)
    for group in config["groups"]:
        bank_path = ROOT / "data/wine/evaluation" / config["label"] / f"{group}.json"
        bank = json.loads(bank_path.read_text())
        digest = sha256_file(bank_path)
        assert len(bank["states"]) == 200
        assert len({entry["sha256"] for entry in bank["states"]}) == 200
        for step in config["checkpoint_steps"]:
            paired = {}
            for modality in config["modalities"]:
                key = f"{modality}-{step}-{group}"
                checkpoint = training["checkpoints"][modality][str(step)]
                assert sha256_file(checkpoint["path"]) == checkpoint["sha256"]
                policy = Student(checkpoint["path"], "cpu")
                output = root / "evaluation" / f"{modality}-{step}" / group
                summary = json.loads((output / "summary.json").read_text())
                records = episode_index(output)
                assert not list(output.glob("*.partial.h5"))
                assert summary["completed"] and summary["completed_attempts"] == 200
                assert summary["policy"] == policy.metadata and summary["evaluation_bank_sha256"] == digest
                assert len(check_records(records, bank, policy.metadata, digest, 4, record_images=False)) == 200
                indexed = {record["initial_state_id"]: record for record in records}
                audits = []
                for entry in bank["states"]:
                    record = indexed[entry["id"]]
                    assert record["config"] == {**bank["config"], "teacher_replan_steps": 4}
                    with h5py.File(record["path"], "r") as stream:
                        assert not stream.attrs["error"] and stream.attrs["schema_version"] == 2
                        np.testing.assert_array_equal(stream["initial_sim_state"][:], entry["state"])
                        assert hashlib.sha256(stream["model_xml"].asstr()[()].encode()).hexdigest() == entry["xml_sha256"]
                        steps = stream["steps"]
                        assert len(steps["phase"]) <= 310
                        if not record["success"]:
                            assert len(steps["phase"]) == 310
                        first = stream["first_policy_observation"]
                        first_index = int(first.attrs["step_index"])
                        observation = {name: first[name][:] for name in
                                       ("state", f"{modality}_agentview", f"{modality}_wrist")}
                        action = policy.infer(observation, record["instruction"])
                        count = min(4, len(steps["phase"]) - first_index)
                        np.testing.assert_allclose(action[:count], steps["executed_action"][first_index:first_index + count],
                                                   atol=1e-6, rtol=0)
                    audits.append(validate_episode(record["path"], bank))
                outcomes = np.asarray([indexed[entry["id"]]["success"] for entry in bank["states"]])
                paired[modality] = outcomes
                successes = int(outcomes.sum())
                assert successes == summary["successes"]
                row = {"distribution": group, "modality": modality, "training_step": step,
                       "attempts": 200, "successes": successes, "timeouts": 200 - successes,
                       "success_rate": successes / 200, "checkpoint_sha256": checkpoint["sha256"]}
                rows.append(row)
                replay_report = directory / f"{key}-replay.json"
                replay_log = directory / f"{key}-replay.log"
                with replay_log.open("w") as log:
                    subprocess.run(["bash", "scripts/sim_cpu.sh", "-m", "depth_policy.replay",
                                    indexed[0]["path"], "--report", str(replay_report)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                replay = json.loads(replay_report.read_text())
                assert replay["passed"] and replay["image_hashes_verified"]
                details[key] = {**row, "bank_sha256": digest, "first_action_chunks_verified": True,
                                "termination_counts": dict(collections.Counter(record["termination_reason"] for record in records)),
                                "success_by_initial_state": outcomes.tolist(), "trajectory_audit": audits,
                                "fixed_id0_replay_audit": replay}
                atomic_json(directory / f"{key}-audit.json", details[key])
                print(json.dumps(row), flush=True)
            pairs[f"{group}-{step}"] = {
                "both_success": int((paired["depth"] & paired["rgb"]).sum()),
                "depth_only_success": int((paired["depth"] & ~paired["rgb"]).sum()),
                "rgb_only_success": int((~paired["depth"] & paired["rgb"]).sum()),
                "both_fail": int((~paired["depth"] & ~paired["rgb"]).sum()),
            }
    return rows, details, pairs


def make_figures(config, rows, details):
    destination = ROOT / "reports/figures" / config["label"]
    destination.mkdir(parents=True, exist_ok=True)
    groups = list(config["groups"])
    titles = {"official-range200": "Original LIBERO placement range (200 fresh states)",
              "expanded10cm200": "Expanded 10 x 10 cm range (200 fresh states)"}
    colors = {"depth": "#2472b4", "rgb": "#e28524"}
    lookup = {(row["distribution"], row["modality"], row["training_step"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, group in zip(axes, groups, strict=True):
        for modality in config["modalities"]:
            rates = [100 * lookup[group, modality, step]["success_rate"] for step in config["checkpoint_steps"]]
            axis.plot(np.asarray(config["checkpoint_steps"]) / 1000, rates, "o-", label=modality.upper(),
                      color=colors[modality], linewidth=2)
            for step, rate in zip(config["checkpoint_steps"], rates, strict=True):
                axis.annotate(f"{rate:g}%", (step / 1000, rate),
                              xytext=(0, 9 if modality == "depth" else -16), textcoords="offset points",
                              ha="center", fontsize=9, color=colors[modality])
        axis.set(title=titles[group], xlabel="Training step (thousands)", ylim=(-5, 110),
                 xticks=np.asarray(config["checkpoint_steps"]) / 1000)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right")
    axes[0].set_ylabel("Success rate (%)")
    figure.suptitle("300 demos | trained from scratch | state + language retained | seed 0")
    figure.tight_layout()
    figure.savefig(destination / "success-curves.png", dpi=170)
    plt.close(figure)
    cells = []
    for step in config["checkpoint_steps"]:
        cells.append([f"{step:,}"] + [f"{lookup[group, modality, step]['successes']}/200 "
                      f"({lookup[group, modality, step]['success_rate']:.1%})"
                      for group in groups for modality in config["modalities"]])
    figure, axis = plt.subplots(figsize=(13, 3.3))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=["Training step", "Original: Depth", "Original: RGB",
                       "10cm: Depth", "10cm: RGB"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    axis.set_title("Successes / 200 paired initial states | 300 training demos", pad=15)
    figure.tight_layout()
    figure.savefig(destination / "success-table.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    for group in groups:
        bank = json.loads((ROOT / "data/wine/evaluation" / config["label"] / f"{group}.json").read_text())
        positions = np.asarray([entry["initial_positions"]["wine_bottle_1"][:2] for entry in bank["states"]])
        positions = (positions - [-0.20, -0.05]) * 100
        figure, axes = plt.subplots(2, 5, figsize=(21, 9), sharex=True, sharey=True)
        for row_index, modality in enumerate(config["modalities"]):
            for column, step in enumerate(config["checkpoint_steps"]):
                axis = axes[row_index, column]
                success = np.asarray(details[f"{modality}-{step}-{group}"]["success_by_initial_state"])
                axis.scatter(positions[success, 0], positions[success, 1], color="#218c54", s=17, label="Success")
                axis.scatter(positions[~success, 0], positions[~success, 1], color="#d24747", s=21, marker="x", label="Timeout")
                span = 5.2 if group == "expanded10cm200" else 1.6
                axis.set(xlim=(-span, span), ylim=(-span, span), aspect="equal",
                         title=f"{modality.upper()} {step // 1000}k: {success.sum()}/200 ({success.mean():.1%})")
                axis.add_patch(Rectangle((-1.5, -1.5), 3, 3, fill=False, linestyle="--", edgecolor="#496a9a"))
                axis.grid(alpha=0.15)
                if row_index == 1:
                    axis.set_xlabel("Bottle x offset (cm)")
                if column == 0:
                    axis.set_ylabel("Bottle y offset (cm)")
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(titles[group] + " | Green: success; red: timeout; dashed: original center range")
        figure.tight_layout()
        figure.savefig(destination / f"{group}-position-maps.png", dpi=150)
        plt.close(figure)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-training-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/wine50k_experiment.json").read_text())
    root = ROOT / "runs" / config["label"]
    training = verify_training(config, root)
    if args.verify_training_only:
        print("All ten checkpoints and training dataset verified", flush=True)
        return
    rows, details, pairs = audit_evaluations(config, root, training)
    figures = make_figures(config, rows, details)
    destination = ROOT / "reports" / config["label"]
    with (destination / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(destination / "results.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                "protocol": config, "training_audit": training, "total_attempts": sum(row["attempts"] for row in rows),
                "attempts_definition": "Completed policy trials; success or task-budget timeout",
                "additional_infrastructure_interrupted_episodes": 5,
                "infrastructure_recovery_report": str(ROOT / "reports/wine50k-startup-recovery.json"),
                "rows": rows, "paired_counts": pairs, "figures": str(figures), "complete": True})
    lookup = {(row["distribution"], row["modality"], row["training_step"]): row for row in rows}
    lines = ["# 300条示范：RGB / Depth 从零训练至50k的配对评估", "",
             "两模型输入分别为双视角RGB/深度 + 8维机器人状态 + 语言，训练seed=0；没有视觉或语言预训练。",
             "训练集固定为原300条成功示范（35个训练初态），各自从随机初始化训练50000 step。",
             "官方原始范围和10cm×10cm范围各生成200个全新独立初态，每种分布全部权重共用。",
             "不是官方50个保存初态重复4次。两组各200个不同状态，20组评估共4000次真实闭环。", "",
             "| 训练step | 原范围Depth | 原范围RGB | 10cm Depth | 10cm RGB |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for step in config["checkpoint_steps"]:
        cells = [f"{lookup[group, modality, step]['successes']}/200 ({lookup[group, modality, step]['success_rate']:.1%})"
                 for group in config["groups"] for modality in config["modalities"]]
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    lines += ["", "## 协议与局限", "",
              "- 每个控制步检查官方On谓词，首次触发立即结束；不额外要求松爪、撤离或持续稳定。",
              "- 20Hz；10步稳定后最多300个任务控制步，超时不是网络或推理故障。每次执行4步后重新推理。",
              "- 10cm指x/y各±5cm；原任务实际中心范围约3cm×3cm。除酒瓶区域外不修改任务。",
              "- 新随机范围测试是本地扩展评估，不等同于官方固定50初态基准。",
              "- 全部固定10k/20k/30k/40k/50k权重均报告，没有按测试结果选择权重或丢弃失败。",
              "- 全部4000条保存动作、状态、时间、图像hash、首个策略观测和初末仿真状态；没有保存每帧完整图像。",
              "- 每条轨迹的首动作块由对应checkpoint重算核验；20组各固定初态0执行精确回放，并校验全部RGB-D hash。",
              "- 两种模态使用相同训练数据、状态与语言分支、优化器和训练预算；视觉输入通道数不同。",
              "- 单任务、单个训练seed，位置图只显示瓶的x/y；其他随机化因素没有在二维图里展示。",
              "- 全部轨迹用于评估，不混入训练数据；教师采集仍暂停。", "",
              "- 另有5条评估启动阶段因共享配置写入竞争而被协调器中断的轨迹，已原样归档；不作为策略超时。",
              "  修复配置原子写入后，使用原冻结初态和权重重跑；没有丢弃已完成的成功或超时结果。",
              "  见 reports/wine50k-startup-recovery.json。", "",
              f"图像目录：{figures}", ""]
    (destination / "results.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "attempts": 4000, "report": str(destination / "results.md")}), flush=True)


if __name__ == "__main__":
    main()
