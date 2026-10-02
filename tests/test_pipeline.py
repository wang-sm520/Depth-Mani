import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from depth_policy.common import orient_image, state_vector, validate_action
from depth_policy.data import EpisodeDataset, encode_language, fit_statistics, preprocess_images
from depth_policy.episodes import EpisodeWriter, episode_index
from depth_policy.manifest import create_manifest
from depth_policy.model import StudentPolicy, masked_action_loss
from depth_policy.train import save_checkpoint


torch.set_num_threads(2)


def test_state_and_orientation():
    observation = {"robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [0, 0, 0, 1],
                   "robot0_gripper_qpos": [0.01, -0.01]}
    np.testing.assert_allclose(state_vector(observation), [1, 2, 3, 0, 0, 0, 0.01, -0.01])
    image = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(orient_image(orient_image(image)), image)
    with pytest.raises(ValueError):
        validate_action([0] * 8)
    with pytest.raises(ValueError):
        validate_action([float("nan")] * 7)


def test_split_groups_duplicates(tmp_path):
    initial = tmp_path / "states.bin"
    initial.write_bytes(b"test")
    states = np.concatenate([np.arange(60).reshape(20, 3), np.arange(60).reshape(20, 3)])
    manifest = create_manifest(states, {"seed": 3, "task": "task", "suite": "suite"}, initial)
    assert manifest["unique_states"] == 20
    for first, second in zip(manifest["states"][:20], manifest["states"][20:], strict=True):
        assert first["sha256"] == second["sha256"]
        assert first["split"] == second["split"]


def make_episode(directory, split="train", count=5, action_value=1.0):
    metadata = {"split": split, "instruction": "put bowl on plate", "seed": 0,
                "initial_state_id": 0 if split == "train" else 1,
                "initial_state_sha256": split,
                "teacher": {"policy_config": "pi05_libero"}}
    writer = EpisodeWriter(directory, metadata, np.zeros(10), "<mujoco/>")
    for step in range(count):
        observation = {"state": np.full(8, step, dtype=np.float32),
                       "depth_agentview": np.ones((16, 16), dtype=np.float32),
                       "depth_wrist": np.ones((16, 16), dtype=np.float32) * 2,
                       "rgb_agentview": np.zeros((16, 16, 3), dtype=np.uint8),
                       "rgb_wrist": np.zeros((16, 16, 3), dtype=np.uint8)}
        writer.append(observation, np.full(7, action_value), step, step * 0.05,
                      np.zeros(10), int(step > 0), np.zeros((2, 12)))
    path = writer.finish(True, "success", np.zeros(10), [0.1])
    return path


def test_episode_atomic_and_unique(tmp_path):
    first = make_episode(tmp_path)
    second = make_episode(tmp_path)
    assert first != second
    assert len(episode_index(tmp_path)) == 2
    assert not list(tmp_path.glob("*.partial.h5"))
    writer = EpisodeWriter(tmp_path, {}, np.zeros(1), "xml")
    writer.file.close()
    assert len(episode_index(tmp_path)) == 2


def test_retained_training_checkpoints(tmp_path):
    from depth_policy.train import save_snapshot

    save_snapshot(tmp_path, {"step": 500}, 10000)
    assert not list(tmp_path.iterdir())
    for step in (10000, 20000, 30000, 40000, 50000):
        save_snapshot(tmp_path, {"step": step}, 10000)
        saved = torch.load(tmp_path / f"step_{step:06d}.pt", weights_only=False)
        assert saved["step"] == step
    with pytest.raises(FileExistsError):
        save_snapshot(tmp_path, {"step": 10000}, 10000)


def test_compact_evaluation_preserves_replay_and_first_input(tmp_path):
    from depth_policy.episodes import IMAGE_FIELDS, image_digest
    from depth_policy.validate import validate_episode

    metadata = {"record_images": False, "evaluation_only": True, "split": "random-test",
                "initial_state_id": 0, "initial_state_sha256": "initial", "depth_units": "m",
                "orientation": "flip_both_axes", "control_freq": 20,
                "config": {"settle_steps": 1, "resolution": 16}}
    with pytest.raises(ValueError):
        EpisodeWriter(tmp_path, {"record_images": False}, np.zeros(10), "xml")
    writer = EpisodeWriter(tmp_path, metadata, np.zeros(10), "xml")
    for index in range(3):
        observation = {"state": np.full(8, index, dtype=np.float32),
                       "depth_agentview": np.full((16, 16), index + 1, dtype=np.float32),
                       "depth_wrist": np.ones((16, 16), dtype=np.float32),
                       "rgb_agentview": np.full((16, 16, 3), index, dtype=np.uint8),
                       "rgb_wrist": np.zeros((16, 16, 3), dtype=np.uint8)}
        writer.append(observation, np.zeros(7), index, index / 20, np.zeros(10),
                      int(index > 0), np.zeros((2, 12)))
    path = writer.finish(True, "success", np.zeros(10), [0.1])
    with h5py.File(path, "r") as stream:
        assert stream.attrs["schema_version"] == 2
        for name in IMAGE_FIELDS:
            assert name not in stream["steps"]
            np.testing.assert_array_equal(stream[f"steps/{name}_sha256"][2], image_digest(observation[name]))
        assert stream["first_policy_observation"].attrs["step_index"] == 1
        assert stream["first_policy_observation/depth_agentview"][0, 0] == 2
    bank = {"states": [{"split": "random-test", "sha256": "initial"}]}
    assert validate_episode(path, bank)["record_images"] is False


def test_full_image_replay_checks_both_rgb_views(tmp_path, monkeypatch):
    from depth_policy import replay

    observation = {"state": np.zeros(8, dtype=np.float32),
                   "depth_agentview": np.ones((16, 16), dtype=np.float32),
                   "depth_wrist": np.ones((16, 16), dtype=np.float32),
                   "rgb_agentview": np.zeros((16, 16, 3), dtype=np.uint8),
                   "rgb_wrist": np.zeros((16, 16, 3), dtype=np.uint8)}
    metadata = {"seed": 10, "config": {}, "instruction": "put cheese on bowl",
                "control_freq": 20}
    writer = EpisodeWriter(tmp_path, metadata, np.zeros(10), "<mujoco/>")
    writer.append(observation, np.zeros(7), 0, 0.0, np.zeros(10), 1, np.zeros((2, 12)))
    path = writer.finish(True, "success", np.zeros(10), [0.0])

    class Environment:
        def __init__(self):
            self.sim = SimpleNamespace(model=SimpleNamespace(get_xml=lambda: "<mujoco/>"),
                                       data=SimpleNamespace(time=0.0))

        def reset(self):
            pass

        def set_init_state(self, state):
            return None

        def get_sim_state(self):
            return np.zeros(10)

        def step(self, action):
            self.sim.data.time += 0.05
            return None, None, None, None

        def check_success(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(replay, "make_env", lambda config, seed: (Environment(), None))
    monkeypatch.setattr(replay, "observe", lambda environment, raw: observation)
    report = tmp_path / "replay.json"
    monkeypatch.setattr("sys.argv", ["replay", str(path), "--report", str(report)])
    replay.main()
    result = json.loads(report.read_text())
    assert result["full_rgbd_verified"] and result["maximum_absolute_errors"]["rgb_wrist"] == 0

    with h5py.File(path, "r+") as stream:
        stream["steps/rgb_wrist"][0, 0, 0, 0] = 1
    with pytest.raises(AssertionError):
        replay.main()


def test_dataset_no_cross_episode_or_settling(tmp_path):
    make_episode(tmp_path, count=5, action_value=1)
    make_episode(tmp_path, count=4, action_value=99)
    episodes = episode_index(tmp_path)
    stats = {"state_mean": [0] * 8, "state_std": [1] * 8,
             "action_mean": [0] * 7, "action_std": [1] * 7}
    dataset = EpisodeDataset(episodes, "depth", {"<pad>": 0, "<unk>": 1}, stats, image_size=16)
    assert len(dataset) == 7
    boundary = dataset.offsets[1] - 1
    example = dataset[boundary]
    assert example["mask"].tolist() == [True] + [False] * 7
    assert torch.count_nonzero(example["action"][1:]) == 0
    assert dataset[0]["state"][0] == 1
    assert dataset[dataset.offsets[1]]["state"][0] == 1
    assert dataset[0]["images"].shape == (2, 2, 16, 16)
    with pytest.raises(IndexError):
        dataset[-1]


def test_normalization_train_only(tmp_path):
    make_episode(tmp_path)
    episodes = episode_index(tmp_path)
    statistics = fit_statistics(episodes)
    np.testing.assert_allclose(statistics["state_mean"], [2.5] * 8)
    episodes[0]["split"] = "test"
    with pytest.raises(ValueError):
        fit_statistics(episodes)


def test_fixed_depth_scale_and_invalid_mask():
    first = np.ones((1, 4, 4), dtype=np.float32)
    first[0, 0, 0] = np.nan
    second = np.ones((1, 4, 4), dtype=np.float32) * 2
    result = preprocess_images(first, second, "depth", size=4, depth_max=3)
    assert result[0, 0, 0, 1, 1] == pytest.approx(1 / 3)
    assert result[0, 1, 0, 1, 1] == pytest.approx(2 / 3)
    assert result[0, 0, 0, 0, 0] == 0 and result[0, 0, 1, 0, 0] == 0
    assert preprocess_images(None, None, "state") is None


@pytest.mark.parametrize("modality,channels", [("depth", 2), ("rgb", 3), ("state", 0)])
def test_model_backward_and_checkpoint(modality, channels, tmp_path):
    model = StudentPolicy(modality, 10)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 5_000_000 <= parameters <= 15_000_000
    images = torch.zeros(2, 2, channels, 32, 32) if channels else None
    state = torch.randn(2, 8)
    tokens = torch.tensor([[1, 2, 0], [2, 1, 0]])
    prediction = model(state, tokens, images)
    assert prediction.shape == (2, 8, 7)
    mask = torch.tensor([[True] * 8, [True] + [False] * 7])
    loss = masked_action_loss(prediction, torch.ones_like(prediction), mask)
    loss.backward()
    assert model.action_head.weight.grad is not None
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"model": model.state_dict()})
    restored = StudentPolicy(modality, 10)
    restored.load_state_dict(torch.load(path, weights_only=True)["model"])
    torch.testing.assert_close(prediction, restored(state, tokens, images))


def test_masked_loss_ignores_padding():
    prediction = torch.zeros(1, 8, 7, requires_grad=True)
    target = torch.ones_like(prediction)
    target[:, 1:] = 1000
    mask = torch.tensor([[True] + [False] * 7])
    loss = masked_action_loss(prediction, target, mask)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, 1:]) == 0


def test_depth_dataset_does_not_read_rgb(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as stream:
        del stream["steps/rgb_agentview"]
        del stream["steps/rgb_wrist"]
    episodes = episode_index(tmp_path)
    dataset = EpisodeDataset(episodes, "depth", {"<pad>": 0, "<unk>": 1},
                             fit_statistics(episodes), image_size=16)
    assert dataset[0]["images"].shape[1] == 2


def test_language_tokens():
    assert encode_language("Put bowl.", {"<pad>": 0, "<unk>": 1, "put": 2, "bowl": 3})[:3].tolist() == [2, 3, 0]
    with pytest.raises(ValueError):
        encode_language("", {})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, device):
    from depth_policy.train import main

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = tmp_path / "data"
    make_episode(data, "train")
    make_episode(data, "validation")
    def train(output, steps, resume=False):
        argv = ["train", "--data", str(data), "--output", str(output), "--modality", "state",
                "--device", device, "--steps", str(steps), "--batch-size", "2",
                "--threads", "2", "--save-every", "1"]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr("sys.argv", argv)
        main()
        return torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    train(tmp_path / "resumed", 1)
    resumed = train(tmp_path / "resumed", 2, resume=True)
    uninterrupted = train(tmp_path / "full", 2)
    assert resumed["step"] == uninterrupted["step"] == 2
    for name, parameter in uninterrupted["model"].items():
        torch.testing.assert_close(parameter, resumed["model"][name], rtol=0, atol=0)


def test_inference_uses_training_preprocessing(tmp_path):
    from depth_policy.evaluate import Student

    model = StudentPolicy("depth", 4)
    statistics = {"state_mean": [1] * 8, "state_std": [2] * 8,
                  "action_mean": [0.1] * 7, "action_std": [0.5] * 7}
    vocabulary = {"<pad>": 0, "<unk>": 1, "put": 2, "bowl": 3}
    config = {"modality": "depth", "horizon": 8, "image_size": 32, "depth_max": 3.0, "seed": 0}
    checkpoint = tmp_path / "student.pt"
    save_checkpoint(checkpoint, {"model": model.state_dict(), "config": config, "vocabulary": vocabulary,
                                 "statistics": statistics, "step": 1})
    observation = {"depth_agentview": np.ones((64, 64), np.float32),
                   "depth_wrist": np.ones((64, 64), np.float32) * 2,
                   "state": np.arange(8, dtype=np.float32)}
    actual = Student(checkpoint, "cpu").infer(observation, "put bowl")
    images = preprocess_images(observation["depth_agentview"][None], observation["depth_wrist"][None],
                               "depth", 32, 3).half()
    state = torch.tensor((observation["state"] - 1) / 2)[None]
    tokens = torch.from_numpy(encode_language("put bowl", vocabulary))[None]
    with torch.no_grad():
        expected = model(state, tokens, images)[0].numpy() * 0.5 + 0.1
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_summary_counts_failed_and_incomplete_attempts(tmp_path):
    from depth_policy.summarize import summarize

    make_episode(tmp_path)
    failed = make_episode(tmp_path)
    with h5py.File(failed, "r+") as stream:
        stream.attrs["success"] = False
        stream.attrs["termination_reason"] = "timeout"
    incomplete = EpisodeWriter(tmp_path, {}, np.zeros(1), "xml")
    incomplete.file.close()
    result = summarize(tmp_path)
    assert result["completed_attempts"] == 2
    assert result["incomplete_attempts"] == 1
    assert result["total_recorded_attempts"] == 3
    assert result["success_rate_all_recorded_attempts"] == pytest.approx(1 / 3)
    assert result["splits"]["train"]["unique_initial_states"] == 1
    assert result["splits"]["train"]["termination_counts"] == {"success": 1, "timeout": 1}
    assert result["inference"]["queries"] == 2
    assert result["inference"]["median_seconds"] == pytest.approx(0.1)


def test_teacher_pilot_gate_requires_coverage_and_clean_attempts():
    import copy
    from depth_policy.experiment import pilot_gate

    report = {"incomplete_attempts": 0, "policies": [{"policy_config": "pi05_libero"}],
              "splits": {"train": {"attempts": 20, "unique_initial_states": 20,
                                   "success_rate": 1.0, "termination_counts": {"success": 20}}}}
    pilot_gate(report)
    for field, value in [("attempts", 19), ("unique_initial_states", 19), ("success_rate", 0.7),
                         ("termination_counts", {"success": 19, "exception": 1})]:
        changed = copy.deepcopy(report)
        changed["splits"]["train"][field] = value
        with pytest.raises(ValueError):
            pilot_gate(changed)
    report["incomplete_attempts"] = 1
    with pytest.raises(ValueError):
        pilot_gate(report)


def test_experiment_routes_task_specific_data(monkeypatch):
    from argparse import Namespace
    from depth_policy.experiment import Experiment

    experiment = Experiment(Namespace(config="configs/wine.json", label="routing-test", target=100, steps=10,
                                     training_python="/test/gpu/python", training_device="cuda"))
    commands = []

    def capture(stage, arguments, timeout):
        commands.append((stage, arguments))
        raise InterruptedError("Stop before executing")

    monkeypatch.setattr(experiment, "command", capture)
    with pytest.raises(InterruptedError):
        experiment.collect()
    validation = commands[-1][1]
    assert validation[validation.index("--config") + 1] == "configs/wine.json"
    assert validation[validation.index("--data") + 1] == "data/wine/teacher"
    with pytest.raises(InterruptedError):
        experiment.train_and_evaluate()
    training = commands[-1][1]
    assert "/test/gpu/python" in training
    assert training[training.index("--device") + 1] == "cuda"
    assert training[training.index("--data") + 1] == "data/wine/teacher"
    assert training[training.index("--output") + 1].startswith(
        "runs/put_the_wine_bottle_on_top_of_the_cabinet/")


def test_pilot_service_accepts_successful_retained_unit(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    from types import SimpleNamespace
    import depth_policy.experiment as module

    report_path = tmp_path / "pilot.json"
    report_path.write_text(json.dumps({
        "directory": "data/wine/teacher", "incomplete_attempts": 0,
        "policies": [{"policy_config": "pi05_libero"}],
        "splits": {"train": {"attempts": 20, "unique_initial_states": 20,
                              "success_rate": 1.0, "termination_counts": {"success": 20}}}}))
    experiment = module.Experiment(Namespace(config="configs/wine.json", label="service-test",
                                             pilot_service="test.service", pilot_report=str(report_path)))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="LoadState=loaded\nActiveState=active\nSubState=exited\nResult=success\nExecMainStatus=0\n"))

    def unexpected_wait(seconds):
        raise AssertionError("A successfully exited retained unit must not keep waiting")

    monkeypatch.setattr(module.time, "sleep", unexpected_wait)
    experiment.wait_for_pilot_service()
    assert "teacher-pilot-gate" in experiment.state["completed_stages"]
