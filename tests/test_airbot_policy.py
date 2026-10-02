from copy import deepcopy
import json

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.data import AirbotDataset, fit_statistics
from airbot_depth.depth import canonical_depth_config
from airbot_depth.model import AirbotPolicyModel, CHECKPOINT_FORMAT, IMAGE_INPUT_PRECISION, batch_loss
from airbot_depth.policy import AirbotDepthPolicy
from airbot_depth.train import evaluate_loss, main as train_main, save_checkpoint
from depth_policy.data import build_vocabulary


torch.set_num_threads(2)
PROMPT = "pick up the red paper bag and hold in the reset position"


def manifest_template():
    config = canonical_depth_config({"image_size": 32})
    return {"schema_version": 1, "status": "complete", "fps": 25, "robot_type": "airbot_play",
            "camera_keys": ["observation.images.head", "observation.images.wrist"],
            "state_dim": 7, "action_dim": 7,
            "state_names": [f"joint_{index}" for index in range(6)] + ["gripper"],
            "action_names": [f"joint_{index}" for index in range(6)] + ["gripper"],
            "action_semantics": "absolute_joint_position", "prompt": PROMPT,
            "depth_config": config, "depth_provenance": {"config": config, "test_fixture": True},
            "episodes": [], "source": {"selected_episode_indices": []}}


def make_checkpoint(path):
    torch.manual_seed(52)
    manifest = manifest_template()
    vocabulary = build_vocabulary([{"instruction": PROMPT}])
    config = {"vocabulary_size": len(vocabulary), "state_dim": 7, "action_dim": 7,
              "views": 2, "horizon": 3}
    model = AirbotPolicyModel(**config)
    statistics = {"state_mean": [0.1] * 7, "state_std": [0.3] * 7,
                  "action_mean": list(np.arange(7, dtype=float)), "action_std": [0.2] * 7}
    payload = {"schema_version": 1, "format": CHECKPOINT_FORMAT,
               "image_input_precision": IMAGE_INPUT_PRECISION,
               "config": {"horizon": 3}, "model_config": config, "model": model.state_dict(),
               "manifest": manifest, "manifest_sha256": "0" * 64, "vocabulary": vocabulary,
               "statistics": statistics, "depth_config": manifest["depth_config"],
               "depth_provenance": manifest["depth_provenance"],
               "action_semantics": "absolute_joint_position"}
    save_checkpoint(path, payload)
    return payload


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "policy.pt"
    make_checkpoint(path)
    return path


def depth_example(views=2, size=32):
    result = np.zeros((views, 2, size, size), dtype=np.float32)
    result[:, 0] = np.linspace(0, 1, size * size, dtype=np.float32).reshape(size, size)
    result[:, 1] = 1
    return result


def test_model_uses_dataset_dimensions_and_masked_backward():
    model = AirbotPolicyModel(5, state_dim=6, action_dim=4, views=3, horizon=3)
    batch = {"state": torch.randn(2, 6), "tokens": torch.tensor([[2, 3, 0], [4, 0, 0]]),
             "images": torch.rand(2, 3, 2, 32, 32), "action": torch.ones(2, 3, 4),
             "mask": torch.tensor([[True, True, True], [True, False, False]])}
    prediction = model(batch["state"], batch["tokens"], batch["images"])
    assert prediction.shape == (2, 3, 4)
    loss = batch_loss(model, batch)
    loss.backward()
    assert torch.isfinite(model.action_head.weight.grad).all()
    batch["action"][1, 1:] = 9000
    torch.testing.assert_close(loss, batch_loss(model, batch))
    with pytest.raises(ValueError, match="state"):
        model(torch.zeros(2, 8), batch["tokens"], batch["images"])
    with pytest.raises(ValueError, match="depth"):
        model(batch["state"], batch["tokens"], batch["images"][:, :2])


def test_depth_policy_undoes_normalization_as_absolute_positions(checkpoint):
    policy = AirbotDepthPolicy(checkpoint)
    with torch.no_grad():
        policy.model.action_head.weight.zero_()
        policy.model.action_head.bias.fill_(0.5)
    actions = policy.infer_depth(np.full(7, 20.0), depth_example())
    # Absolute outputs do not add the robot's current joint position.
    np.testing.assert_allclose(actions, np.broadcast_to(np.arange(7) + 0.1, (3, 7)), atol=1e-6)
    assert actions.dtype == np.float32
    assert policy._depth_transform is None


def test_rgb_camera_order_and_online_offline_input_match(checkpoint, monkeypatch):
    import airbot_depth.policy as policy_module

    policy = AirbotDepthPolicy(checkpoint)
    depth = depth_example()
    captured = []

    class FakeTransform:
        def __init__(self, config, device, local_files_only):
            self.config = deepcopy(config)

        def provenance(self):
            return deepcopy(policy.depth_provenance)

        def __call__(self, images):
            captured.append([int(image[0, 0, 0]) for image in images])
            return depth.copy()

    monkeypatch.setattr(policy_module, "DepthAnythingTransform", FakeTransform)
    rgb = {policy.camera_keys[1]: np.full((8, 12, 3), 23, dtype=np.uint8),
           policy.camera_keys[0]: np.full((10, 20, 3), 11, dtype=np.uint8)}
    state = np.linspace(0, 1, 7, dtype=np.float32)
    offline = policy.infer_depth(state, depth)
    online = policy.infer_rgb(state, rgb)
    np.testing.assert_array_equal(offline, online)
    assert captured == [[11, 23]]
    # Cached training depth is float16; online float32 receives identical rounding.
    np.testing.assert_array_equal(offline, policy.infer_depth(state, depth.astype(np.float16)))
    with pytest.raises(ValueError, match="camera keys"):
        policy.infer_rgb(state, {policy.camera_keys[0]: rgb[policy.camera_keys[0]]})
    with pytest.raises(ValueError, match="uint8"):
        policy.infer_rgb(state, {key: value.astype(np.float32) for key, value in rgb.items()})


def test_runtime_depth_provenance_mismatch_fails_before_inference(checkpoint, monkeypatch):
    import airbot_depth.policy as policy_module

    policy = AirbotDepthPolicy(checkpoint)

    class WrongTransform:
        def __init__(self, config, device, local_files_only):
            self.config = config

        def provenance(self):
            return {"different_model_hash": "wrong"}

        def __call__(self, images):
            pytest.fail("Inference must not run with a different depth encoder")

    monkeypatch.setattr(policy_module, "DepthAnythingTransform", WrongTransform)
    with pytest.raises(ValueError, match="artifacts/preprocessing differ"):
        policy.infer_rgb(np.zeros(7), {key: np.zeros((8, 8, 3), dtype=np.uint8)
                                       for key in policy.camera_keys})
    assert policy._depth_transform is None


def test_policy_rejects_invalid_observations_and_untrained_prompt(checkpoint):
    policy = AirbotDepthPolicy(checkpoint)
    depth = depth_example()
    with pytest.raises(ValueError, match="exact trained prompt"):
        policy.infer_depth(np.zeros(7), depth, prompt="pick up the blue paper bag")
    with pytest.raises(ValueError, match="finite state"):
        policy.infer_depth(np.zeros(8), depth)
    with pytest.raises(ValueError, match="finite state"):
        policy.infer_depth(np.full(7, np.nan), depth)
    invalid = depth.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite preprocessed depth"):
        policy.infer_depth(np.zeros(7), invalid)
    invalid = depth.copy()
    invalid[0, 1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="validity channel"):
        policy.infer_depth(np.zeros(7), invalid)
    with pytest.raises(ValueError, match="finite preprocessed depth"):
        policy.infer_depth(np.zeros(7), depth[:, :, :16])


@pytest.mark.parametrize("field", ["model_config", "statistics", "depth_config", "action_semantics", "image_input_precision"])
def test_checkpoint_mismatch_is_rejected(tmp_path, field):
    path = tmp_path / "wrong.pt"
    payload = make_checkpoint(path)
    if field == "model_config":
        payload[field]["state_dim"] = 8
    elif field == "statistics":
        payload[field]["state_std"][0] = 0
    elif field == "depth_config":
        payload[field] = {**payload[field], "image_size": 64}
    else:
        payload[field] = "wrong"
    save_checkpoint(path, payload)
    with pytest.raises(ValueError):
        AirbotDepthPolicy(path)


def make_dataset(directory):
    directory.mkdir(exist_ok=True)
    manifest = manifest_template()
    for index, (split, count, value) in enumerate([("train", 3, 1), ("train", 2, 5), ("validation", 2, 99)]):
        path = directory / f"episode_{index}.h5"
        with h5py.File(path, "w") as stream:
            stream.attrs["episode_index"] = index
            stream.attrs["instruction"] = PROMPT
            stream.create_dataset("state", data=np.full((count, 7), value, dtype=np.float32))
            stream.create_dataset("action", data=np.full((count, 7), value, dtype=np.float32))
            stream.create_dataset("depth", data=np.broadcast_to(depth_example(), (count, 2, 2, 32, 32)))
        manifest["episodes"].append({"episode_index": index, "frames": count, "path": path.name,
                                     "sha256": sha256_file(path), "instruction": PROMPT, "split": split})
        manifest["source"]["selected_episode_indices"].append(index)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def test_action_chunks_do_not_cross_episode_boundaries_and_statistics_use_train_only(tmp_path):
    manifest = make_dataset(tmp_path)
    training = manifest["episodes"][:2]
    statistics = fit_statistics(tmp_path, training)
    np.testing.assert_allclose(statistics["state_mean"], [2.6] * 7)
    with pytest.raises(ValueError, match="training episodes only"):
        fit_statistics(tmp_path, manifest["episodes"])
    raw_statistics = {"state_mean": [0] * 7, "state_std": [1] * 7,
                      "action_mean": [0] * 7, "action_std": [1] * 7}
    dataset = AirbotDataset(tmp_path, training, raw_statistics,
                           build_vocabulary([{"instruction": PROMPT}]), horizon=4)
    assert dataset[0]["mask"].tolist() == [True, True, True, False]
    assert dataset[2]["mask"].tolist() == [True, False, False, False]
    assert dataset[3]["mask"].tolist() == [True, True, False, False]
    assert dataset[4]["mask"].tolist() == [True, False, False, False]
    assert torch.count_nonzero(dataset[2]["action"][1:]) == 0
    assert dataset[3]["action"][0].tolist() == [5] * 7
    assert dataset[0]["images"].dtype == torch.float16


def test_validation_loss_is_weighted_by_valid_action_labels():
    class ZeroPolicy(nn.Module):
        def forward(self, state, tokens, images):
            return torch.zeros(len(state), 3, 1)

    dataset = [{"state": torch.zeros(7), "tokens": torch.ones(2, dtype=torch.long),
                "images": torch.zeros(2, 2, 16, 16),
                "action": torch.tensor([[value], [value], [value]]),
                "mask": torch.tensor(mask)}
               for value, mask in [(1.0, [True, True, True]), (3.0, [True, False, False])]]
    model = ZeroPolicy()
    # Smooth-L1(1)=0.5 for three labels; Smooth-L1(3)=2.5 for one label.
    assert evaluate_loss(model, dataset, "cpu", batch_size=1) == pytest.approx(1.0)
    assert evaluate_loss(model, dataset, "cpu", batch_size=2) == pytest.approx(1.0)
    assert model.training


def test_tiny_training_saves_auditable_checkpoint_and_refuses_overwrite(tmp_path):
    data, output = tmp_path / "data", tmp_path / "run"
    manifest = make_dataset(data)
    argv = ["--data", str(data), "--output", str(output), "--steps", "2", "--batch-size", "2",
            "--horizon", "3", "--checkpoint-every", "2", "--eval-every", "2", "--threads", "2"]
    train_main(argv)
    assert (output / "step_000002.pt").is_file()
    assert (output / "best.pt").is_file() and (output / "latest.pt").is_file()
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete" and run["completed_steps"] == 2
    assert run["initialization"] == "all_policy_parameters_random"
    assert run["manifest"] == manifest
    assert run["manifest_sha256"] == sha256_file(data / "manifest.json")
    assert run["source"]["depth_policy/model.py"]
    assert run["environment"]["torch"] == str(torch.__version__)
    policy = AirbotDepthPolicy(output / "step_000002.pt")
    assert policy.infer_depth(np.ones(7), depth_example()).shape == (3, 7)
    with pytest.raises(FileExistsError, match="nonempty run directory"):
        train_main(argv)
