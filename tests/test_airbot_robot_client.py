from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from airbot_deploy import client


ROOT = Path(__file__).resolve().parents[1]
RESET = np.asarray([0, -0.5, 0.4, 0, 0, 0, 0.02], dtype=np.float64)


@pytest.fixture
def profile():
    result = client.load_profile(ROOT / "configs/airbot_paperbag_deploy.json")
    result["transport"]["connect_timeout_s"] = 1.0
    result["transport"]["inference_timeout_s"] = 1.0
    return result


def metadata(profile):
    config = {"representation": "relative_inverse_depth", "input_color": "RGB", "revision": "a" * 40,
              "output_channels": ["relative_inverse_depth", "validity_mask"]}
    provenance = {"config": config, "files_sha256": {key: "b" * 64 for key in
                                                    ("config.json", "preprocessor_config.json", "model.safetensors")},
                  "implementation_sha256": "c" * 64, "libraries": {"numpy": np.__version__},
                  "processor": {"class": "DPTImageProcessor"}}
    return {"schema_version": 1, "policy_type": "airbot_depth", "protocol": "openpi_msgpack_numpy",
            **deepcopy(profile["policy"]), "max_message_bytes": profile["transport"]["max_message_bytes"],
            "preprocessing": {"depth_config": config, "depth_provenance": provenance,
                              "model_input_quantization": "float32_to_float16_to_float32"}}


@contextmanager
def websocket_server(profile, *, response=None, change_metadata=None, delay=0.0):
    hello = metadata(profile)
    if change_metadata is not None:
        change_metadata(hello)
    received = []
    def handler(connection):
        try:
            connection.send(msgpack_numpy.packb(hello))
            while True:
                message = connection.recv()
                observation = msgpack_numpy.unpackb(message)
                received.append(observation)
                if delay:
                    time.sleep(delay)
                actions = np.repeat(RESET[None].astype(np.float32), 8, axis=0)
                actions[:, 0] += np.arange(1, 9) * 0.001
                if response is not None:
                    actions = response(actions)
                connection.send(msgpack_numpy.packb({"actions": actions, "server_timing": {"infer_ms": 1.0}}))
        except ConnectionClosed:
            pass
    with serve(handler, "127.0.0.1", 0, compression=None, max_size=32 * 1024 * 1024) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1], received
        finally:
            server.shutdown()
            thread.join(timeout=2)


class FakeOperator:
    def __init__(self, config):
        self.config = config
        self.position = RESET.copy()
        self.valid = True
        self.actions, self.resets, self.modes, self.speeds = [], [], [], []
        self.shutdown_calls = 0
        self.fail_on_send = None
        self.stale_timestamp = None
        self.images = {}
        for name, width, height in zip(config.camera_names, config.camera_widths, config.camera_heights, strict=True):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0], image[..., 1], image[..., 2] = 19, 87, 231
            self.images[name] = image
        self.robots = {"left": SimpleNamespace(_is_connected=True, _control_error=None,
                                               arm=SimpleNamespace(state=lambda: SimpleNamespace(
                                                   is_valid=self.valid, pos=self.position.copy())))}

    def capture_observation(self):
        timestamp = self.stale_timestamp if self.stale_timestamp is not None else time.time_ns()
        return {"position": self.position.copy(), **{
            f"{name}/color/image_raw": {"data": image, "t": timestamp} for name, image in self.images.items()}}

    def get_qpos(self, observation):
        return observation["position"]

    def switch_mode(self, mode):
        self.modes.append(mode)

    def set_speed_profile(self, mode):
        self.speeds.append(mode)

    def move_to_joint_pos(self, action):
        self.resets.append(action.copy())
        self.position = action.copy()

    def send_action(self, action):
        self.actions.append(action.copy())
        self.position = action.copy()
        if self.fail_on_send is not None:
            raise self.fail_on_send

    def shutdown(self):
        self.shutdown_calls += 1
        return True


def fake_hardware(instances, customize=None):
    def construct(config):
        operator = FakeOperator(config)
        if customize:
            customize(operator)
        instances.append(operator)
        return operator
    return lambda root: SimpleNamespace(RobotAH=construct, RobotAHConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                                        SystemMode=SimpleNamespace(RESETTING="resetting", SAMPLING="sampling"),
                                        SpeedProfile=SimpleNamespace(SLOW="slow", FAST="fast"))


def arguments(port=8026, execute=True, max_steps=5):
    return SimpleNamespace(host="127.0.0.1", port=port, openpi_root="unused-fake-upstream", can_interface="can-test",
                           reset_action=RESET.tolist(), max_steps=max_steps, chunk_size_execute=4, execute=execute)


def test_dry_run_never_imports_hardware_or_connects_network(profile, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run imported/constructed hardware or opened the network")
    client.robot_session(arguments(execute=False), profile, hardware_loader=forbidden, connection_factory=forbidden)
    result = json.loads(capsys.readouterr().out)
    assert result["hardware_imported"] is False and result["execute"] is False
    assert result["checkpoint_sha256"] == profile["policy"]["checkpoint_sha256"]
    assert result["control_frequency"] == 100 and result["fps"] == 25


@pytest.mark.parametrize("frequency", ["missing", None, True, 100.0, "100", 0, 25, 250])
def test_profile_requires_explicit_verified_control_frequency(profile, tmp_path, frequency):
    if frequency == "missing":
        profile["robot"].pop("control_frequency", None)
    else:
        profile["robot"]["control_frequency"] = frequency
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="100 Hz low-level control frequency"):
        client.load_profile(path)


def test_import_is_numpy_only_without_torch_jax_or_robot_sdk():
    command = "import sys; import airbot_deploy.client; assert not ({'torch','jax','airbot_hardware_py','play_operator_ah'} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True, timeout=10)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "prompt", "fps", "camera_keys", "action_units"])
def test_wrong_server_metadata_prevents_even_hardware_construction(profile, field):
    def change(value):
        if field == "camera_keys":
            value[field].reverse()
        elif field == "fps":
            value[field] = 30
        elif field == "action_units":
            value[field][-1] = "rad"
        else:
            value[field] = "wrong"
    def forbidden(*args, **kwargs):
        pytest.fail("Hardware imported before metadata verification")
    with websocket_server(profile, change_metadata=change) as (port, received):
        with pytest.raises(ValueError, match="metadata mismatch"):
            client.robot_session(arguments(port), profile, hardware_loader=forbidden)
    assert not received


def test_normal_chunks_reobserve_preserve_rgb_and_quit_without_extra_reset(profile):
    instances = []
    answers = iter(["", "q"])
    with websocket_server(profile) as (port, received):
        client.robot_session(arguments(port), profile, hardware_loader=fake_hardware(instances),
                             input_fn=lambda prompt: next(answers))
    operator = instances[0]
    assert len(operator.actions) == 5 and len(operator.resets) == 1
    assert operator.shutdown_calls == 1 and len(received) == 2
    assert operator.modes == ["resetting", "sampling"]
    for request in received:
        assert request["prompt"] == profile["policy"]["prompt"]
        for name, image in operator.images.items():
            np.testing.assert_array_equal(request[f"observation/{name}"], image)
    assert received[0]["observation/base_0_rgb"].shape == (1080, 1920, 3)
    assert received[0]["observation/left_wrist_0_rgb"].shape == (480, 848, 3)


@pytest.mark.parametrize("problem", ["joint_limit", "nonfinite", "shape"])
def test_invalid_response_never_sends_any_policy_action(profile, problem):
    instances = []
    def response(actions):
        if problem == "joint_limit":
            actions[3, 1] = 0.18
        elif problem == "nonfinite":
            actions[7, 1] = np.nan
        else:
            actions = actions[:7]
        return actions
    with websocket_server(profile, response=response) as (port, received):
        with pytest.raises(ValueError):
            client.robot_session(arguments(port), profile, hardware_loader=fake_hardware(instances), input_fn=lambda prompt: "")
    # The operator explicitly requested this reset with Enter before inference.
    assert len(instances[0].resets) == 1
    assert not instances[0].actions and instances[0].shutdown_calls == 1
    assert len(received) == 1


def test_request_timeout_closes_connection_without_replaying_actions(profile):
    profile["transport"]["inference_timeout_s"] = 0.05
    instances = []
    with websocket_server(profile, delay=0.2) as (port, received):
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            client.robot_session(arguments(port), profile, hardware_loader=fake_hardware(instances), input_fn=lambda prompt: "")
        assert time.monotonic() - start < 1.5
    assert not instances[0].actions and instances[0].shutdown_calls == 1
    assert len(received) == 1


def test_total_deadline_interrupts_a_blocked_send(profile):
    profile["transport"]["inference_timeout_s"] = 0.02
    released = threading.Event()
    class BlockedConnection:
        def __init__(self):
            self.socket = SimpleNamespace(shutdown=lambda how: released.set())
        def send(self, message):
            assert released.wait(1)
            raise OSError("shutdown interrupted write")
        def close(self):
            released.set()
    connection = object.__new__(client.PolicyConnection)
    connection.profile, connection.codec, connection.socket = profile, msgpack_numpy, BlockedConnection()
    with pytest.raises(TimeoutError, match="timed out"):
        connection.infer({"test": True})
    assert connection.socket is None
    with pytest.raises(RuntimeError, match="closed"):
        connection.infer({"test": True})


def test_gripper_clipping_is_explicit_and_reset_remains_strict(caplog):
    actions = np.repeat(RESET[None], 2, axis=0)
    actions[:, 6] = [-0.001, 0.09]
    commands = client.prepare_actions(actions, episode=2, step=4)
    assert commands[:, 6].tolist() == [0.0, 0.072]
    assert "raw=-0.001000000 m command=0.000000000 m" in caplog.text
    assert "raw=0.090000000 m command=0.072000000 m" in caplog.text
    np.testing.assert_array_equal(actions[:, 6], [-0.001, 0.09])
    with pytest.raises(ValueError, match="Reset gripper"):
        client.prepare_actions(actions, reset=True)


def test_small_negative_measured_gripper_is_clipped_in_interpolated_waypoints(profile, caplog):
    config = SimpleNamespace(**{key: profile["robot"][key] for key in
                                ("camera_names", "camera_widths", "camera_heights")})
    operator = FakeOperator(config)
    operator.position[6] = -0.000004
    action = RESET.copy()
    action[0], action[6] = 0.05, 0.0
    connection = SimpleNamespace(infer=lambda observation: np.repeat(action[None], 8, axis=0))
    client.run_episode(operator, connection, profile, max_steps=1, chunk_size=4, episode=1, previous_stamps={})
    assert len(operator.actions) == 5
    assert all(value[6] >= 0 for value in operator.actions)
    assert "G2 clipping" in caplog.text


def test_invalid_sdk_state_prevents_reset_or_policy_actions(profile):
    instances = []
    with websocket_server(profile) as (port, received):
        with pytest.raises(RuntimeError, match="invalid robot state"):
            client.robot_session(arguments(port), profile,
                                 hardware_loader=fake_hardware(instances, lambda operator: setattr(operator, "valid", False)),
                                 input_fn=lambda prompt: "")
    assert not instances[0].actions and not instances[0].resets
    assert instances[0].shutdown_calls == 1 and not received


def test_camera_timestamps_must_advance_independently(profile):
    profile["robot"]["max_camera_age_s"] = 0.03
    config = SimpleNamespace(**{key: profile["robot"][key] for key in
                                ("camera_names", "camera_widths", "camera_heights")})
    operator = FakeOperator(config)
    stamp = time.time_ns()
    operator.stale_timestamp = stamp
    with pytest.raises(TimeoutError, match="stale|deadline"):
        client.capture_policy_observation(operator, profile, {"base_0_rgb": stamp, "left_wrist_0_rgb": stamp - 1})
    assert not operator.actions


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), client.SignalStop(signal.SIGTERM)])
def test_stop_during_send_cleans_up_without_reset(profile, interruption):
    instances = []
    with websocket_server(profile) as (port, received):
        with pytest.raises(type(interruption)):
            client.robot_session(arguments(port), profile, input_fn=lambda prompt: "",
                                 hardware_loader=fake_hardware(instances, lambda operator: setattr(operator, "fail_on_send", interruption)))
    assert len(instances[0].actions) == 1
    assert len(instances[0].resets) == 1 and instances[0].shutdown_calls == 1
    assert len(received) == 1


def test_q_before_enter_never_resets(profile):
    instances = []
    with websocket_server(profile) as (port, received):
        client.robot_session(arguments(port), profile, hardware_loader=fake_hardware(instances), input_fn=lambda prompt: "q")
    assert not instances[0].resets and not instances[0].actions and not received
    assert instances[0].shutdown_calls == 1


def test_legacy_helper_cannot_silently_drop_the_control_frequency(profile):
    instances = []
    bindings = fake_hardware(instances)(None)

    def legacy_config(**kwargs):
        kwargs.pop("control_frequency", None)
        return SimpleNamespace(**kwargs)

    bindings.RobotAHConfig = legacy_config
    with websocket_server(profile) as (port, received):
        with pytest.raises(ValueError, match="control frequency"):
            client.robot_session(arguments(port), profile, hardware_loader=lambda root: bindings,
                                 input_fn=lambda prompt: "q")
    assert not instances and not received


def test_probe_reference_is_applied_only_to_exact_observation_and_checkpoint(tmp_path):
    observation = tmp_path / "sample.npz"
    np.savez(observation, state=RESET)
    actions = np.repeat(RESET[None], 8, axis=0)
    reference = {"actions": actions.tolist(), "checkpoint_sha256": "a" * 64,
                 "sample_observation_sha256": client.sha256_file(observation)}
    path = tmp_path / "reference_actions.json"
    path.write_text(json.dumps(reference))
    result = client.compare_reference(observation, actions, "a" * 64)
    assert result["status"] == "passed" and result["max_abs_difference"] == 0
    with pytest.raises(ValueError, match="different checkpoint"):
        client.compare_reference(observation, actions, "b" * 64)
    reference["sample_observation_sha256"] = "changed"
    path.write_text(json.dumps(reference))
    assert client.compare_reference(observation, actions, "a" * 64)["status"] == "skipped"
