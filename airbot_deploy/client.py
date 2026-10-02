"""Manual, bounded AIRBOT client for the pinned RGB-to-depth policy service."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import queue
import re
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np


LOGGER = logging.getLogger("airbot_deploy")
JOINT_LIMITS = np.asarray([[-3.14, 2.09], [-2.96, 0.17], [-0.087, 3.14],
                           [-3.01, 3.01], [-1.76, 1.76], [-3.01, 3.01]], dtype=np.float64)
GRIPPER_LIMITS = (0.0, 0.072)
STEP_LENGTH = np.asarray([0.01] * 6 + [0.005], dtype=np.float64)
NAMES = [f"joint{index}.pos" for index in range(1, 7)] + ["eef.pos"]
UNITS = ["rad"] * 6 + ["m"]
CAMERA_KEYS = ["observation.images.head", "observation.images.wrist"]
CAMERA_MAPPING = {"observation/base_0_rgb": CAMERA_KEYS[0], "observation/left_wrist_0_rgb": CAMERA_KEYS[1]}


def emit(event, **fields):
    LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported deployment profile")
    policy, robot, transport = [profile[key] for key in ("policy", "robot", "transport")]
    expected = {"fps": 25, "horizon": 8, "state_dim": 7, "action_dim": 7,
                "state_names": NAMES, "action_names": NAMES, "state_units": UNITS, "action_units": UNITS,
                "action_semantics": "absolute_joint_position", "input_color": "RGB",
                "camera_keys": CAMERA_KEYS, "camera_mapping": CAMERA_MAPPING}
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Deployment profile differs from the verified AIRBOT policy contract")
    if re.fullmatch(r"[0-9a-f]{64}", policy.get("checkpoint_sha256", "")) is None or not policy.get("prompt"):
        raise ValueError("Profile requires its fixed checkpoint SHA256 and prompt")
    if (not np.array_equal(robot["joint_limits"], JOINT_LIMITS)
            or tuple(robot["gripper_limits"]) != GRIPPER_LIMITS
            or robot["gripper_out_of_range"] != "clip_with_warning"
            or not np.array_equal(robot["step_length"], STEP_LENGTH)):
        raise ValueError("Profile must preserve upstream joint/gripper limits and interpolation increments")
    if (robot["camera_names"] != ["base_0_rgb", "left_wrist_0_rgb"]
            or robot["camera_widths"] != [1920, 848] or robot["camera_heights"] != [1080, 480]
            or robot["camera_fps"] != [25, 30] or len(robot["camera_index"]) != 2):
        raise ValueError("Profile must use the recorded native head and wrist camera geometry")
    if type(robot.get("control_frequency")) is not int or robot["control_frequency"] != 100:
        raise ValueError("Profile must use the verified 100 Hz low-level control frequency")
    for value in (robot["max_camera_age_s"], transport["connect_timeout_s"], transport["inference_timeout_s"]):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("Timeouts must be finite and positive")
    if transport["max_message_bytes"] != 32 * 1024 * 1024:
        raise ValueError("Profile must preserve the server's 32 MiB message limit")
    return profile


def validate_metadata(metadata, profile):
    expected = {"schema_version": 1, "policy_type": "airbot_depth", "protocol": "openpi_msgpack_numpy",
                **profile["policy"], "max_message_bytes": profile["transport"]["max_message_bytes"]}
    if not isinstance(metadata, dict) or set(metadata) != {*expected, "preprocessing"}:
        raise ValueError("Server metadata has missing or unknown contract fields")
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Server metadata mismatch: {name}")
    preprocessing = metadata["preprocessing"]
    if not isinstance(preprocessing, dict) or set(preprocessing) != {"depth_config", "depth_provenance", "model_input_quantization"}:
        raise ValueError("Server lacks complete frozen depth preprocessing metadata")
    config, provenance = preprocessing["depth_config"], preprocessing["depth_provenance"]
    if (not isinstance(config, dict) or not isinstance(provenance, dict) or provenance.get("config") != config
            or config.get("representation") != "relative_inverse_depth" or config.get("input_color") != "RGB"
            or config.get("output_channels") != ["relative_inverse_depth", "validity_mask"]
            or preprocessing["model_input_quantization"] != "float32_to_float16_to_float32"
            or re.fullmatch(r"[0-9a-f]{40}", config.get("revision", "")) is None):
        raise ValueError("Server depth preprocessing provenance is incomplete or inconsistent")
    hashes = provenance.get("files_sha256", {})
    if (set(hashes) != {"config.json", "preprocessor_config.json", "model.safetensors"}
            or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values())
            or re.fullmatch(r"[0-9a-f]{64}", provenance.get("implementation_sha256", "")) is None
            or not provenance.get("libraries") or not provenance.get("processor")):
        raise ValueError("Server depth artifacts/implementation are not fingerprinted")
    return metadata


def validate_state(state):
    values = np.asarray(state)
    if values.shape != (7,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Robot state must contain seven finite native joint/gripper values")
    with np.errstate(over="ignore"):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Robot state is not representable as finite float32")
    return values


def validate_rgb(image, width, height):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (height, width, 3):
        raise ValueError(f"Expected native uint8 RGB shape {(height, width, 3)}; no resizing is performed")
    return image


def validate_actions(actions, horizon=8):
    values = np.asarray(actions)
    if values.shape != (horizon, 7) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"Policy must return finite absolute actions [{horizon},7]")
    return values.astype(np.float64, copy=True)


def prepare_actions(actions, *, reset=False, episode=0, step=0):
    """Validate the entire selected chunk before emitting any of its waypoints."""
    values = np.array(actions, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Action/reset targets must be nonempty finite seven-dimensional rows")
    if np.any(values[:, :6] < JOINT_LIMITS[:, 0]) or np.any(values[:, :6] > JOINT_LIMITS[:, 1]):
        raise ValueError("Entire action chunk rejected: joint target exceeds unchanged AIRBOT limits")
    invalid = (values[:, 6] < GRIPPER_LIMITS[0]) | (values[:, 6] > GRIPPER_LIMITS[1])
    if reset and invalid.any():
        raise ValueError("Reset gripper target must already be in [0,0.072] metres")
    for index in np.flatnonzero(invalid):
        raw = float(values[index, 6])
        values[index, 6] = np.clip(raw, *GRIPPER_LIMITS)
        LOGGER.warning("G2 clipping: episode=%d chunk_start_policy_step=%d row=%d raw=%.9f m command=%.9f m",
                       episode, step, int(index), raw, values[index, 6])
    return values


def interpolate_action(previous, action):
    count = int(np.max(np.ceil(np.abs(action - previous) / STEP_LENGTH)))
    return action[None].copy() if count <= 1 else np.linspace(previous, action, count + 1)[1:]


class PolicyConnection:
    """A failed/expired request permanently closes this connection; there is no retry."""

    def __init__(self, profile, host="127.0.0.1", port=8026):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        if not 1 <= port <= 65535 or "/" in host or not host:
            raise ValueError("Invalid inference server host/port")
        self.profile, self.codec = profile, msgpack_numpy
        self.socket = None
        address = f"[{host}]" if ":" in host and not host.startswith("[") else host
        timeout = profile["transport"]["connect_timeout_s"]
        try:
            self.socket = connect(f"ws://{address}:{port}", open_timeout=timeout, close_timeout=1,
                                  compression=None, max_size=profile["transport"]["max_message_bytes"])
            message = self.socket.recv(timeout=timeout)
            if not isinstance(message, bytes):
                raise ValueError("Server metadata must be binary msgpack")
            self.metadata = validate_metadata(msgpack_numpy.unpackb(message), profile)
        except BaseException:
            self.close()
            raise
        emit("policy_verified", host=host, port=port, checkpoint_sha256=self.metadata["checkpoint_sha256"],
             prompt=self.metadata["prompt"], fps=self.metadata["fps"], horizon=self.metadata["horizon"])

    def infer(self, observation):
        if self.socket is None:
            raise RuntimeError("Inference connection is closed; stale actions cannot be replayed")
        connection = self.socket
        timeout = self.profile["transport"]["inference_timeout_s"]
        expired = threading.Event()
        def abort():
            expired.set()
            try:
                connection.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer = threading.Timer(timeout, abort)
        timer.daemon = True
        started = time.monotonic()
        timer.start()
        try:
            connection.send(self.codec.packb(observation))
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Inference send exceeded the request deadline")
            message = connection.recv(timeout=remaining)
            if expired.is_set():
                raise TimeoutError("Inference request exceeded its deadline")
            if not isinstance(message, bytes):
                raise RuntimeError(f"Server rejected inference: {str(message)[:1000]}")
            result = self.codec.unpackb(message)
            if not isinstance(result, dict) or "actions" not in result or set(result) - {"actions", "server_timing"}:
                raise ValueError("Invalid inference response fields")
            actions = validate_actions(result["actions"], self.profile["policy"]["horizon"])
            if expired.is_set() or time.monotonic() - started > timeout:
                raise TimeoutError("Inference decoding exceeded its request deadline")
            return actions
        except BaseException as error:
            timer.cancel()
            self.close()
            if expired.is_set() or isinstance(error, TimeoutError):
                raise TimeoutError("Inference timed out; connection closed and no old action will be resent") from error
            raise
        finally:
            timer.cancel()

    def close(self):
        connection, self.socket = self.socket, None
        if connection is not None:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def _upstream_module(openpi_root, name):
    directory = (Path(openpi_root).resolve() / "examples" / "airbot")
    if not (directory / f"{name}.py").is_file():
        raise FileNotFoundError(directory / f"{name}.py")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != directory:
        raise ValueError(f"Refusing to mix another OpenPI hardware module: {name}")
    return module


def load_hardware(openpi_root):
    operator = _upstream_module(openpi_root, "play_operator_ah")
    config = _upstream_module(openpi_root, "robot_config")
    for name in ("airbot_arm", "camera_sources", "rtsp_camera"):
        _upstream_module(openpi_root, name)
    if not np.array_equal(operator._JOINT_LIMITS, JOINT_LIMITS) or tuple(operator._G2_LIMITS) != GRIPPER_LIMITS:
        raise ValueError("Upstream controller limits differ from the reviewed deployment profile")
    return SimpleNamespace(RobotAH=operator.RobotAH, RobotAHConfig=config.RobotAHConfig,
                           SystemMode=operator.SystemMode, SpeedProfile=operator.SpeedProfile)


def ensure_robot_healthy(operator):
    if set(operator.robots) != {"left"}:
        raise RuntimeError("Expected one connected left AIRBOT Play arm")
    robot = operator.robots["left"]
    if robot._control_error is not None or not robot._is_connected:
        raise RuntimeError("AIRBOT control loop failed or disconnected")
    state = robot.arm.state()
    if not state.is_valid:
        raise RuntimeError("AIRBOT SDK reports invalid robot state")
    return validate_state(state.pos)


def capture_before_deadline(capture, timeout):
    results = queue.Queue(maxsize=1)
    def work():
        try:
            results.put((True, capture()))
        except BaseException as error:
            results.put((False, error))
    threading.Thread(target=work, name="airbot-observation-read", daemon=True).start()
    try:
        ok, result = results.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError("Camera/state capture exceeded its deadline; refusing further actions") from error
    if not ok:
        raise result
    return result


def fresh_camera_images(observation, profile, previous):
    robot = profile["robot"]
    images, stamps = {}, {}
    now = time.time_ns()
    for name, width, height in zip(robot["camera_names"], robot["camera_widths"], robot["camera_heights"], strict=True):
        frame = observation[f"{name}/color/image_raw"]
        timestamp = frame["t"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, np.integer)) or timestamp <= 0:
            raise ValueError(f"{name} requires a positive nanosecond frame timestamp")
        age = (now - int(timestamp)) / 1e9
        if age < -0.1 or age > robot["max_camera_age_s"]:
            raise TimeoutError(f"{name} camera frame age is invalid/stale: {age:.3f}s")
        if name in previous and int(timestamp) <= previous[name]:
            return None
        images[f"observation/{name}"] = validate_rgb(frame["data"], width, height)
        stamps[name] = int(timestamp)
    return images, stamps


def capture_policy_observation(operator, profile, previous):
    deadline = time.monotonic() + profile["robot"]["max_camera_age_s"]
    while True:
        ensure_robot_healthy(operator)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Both camera timestamps did not advance before the capture deadline")
        observation = capture_before_deadline(operator.capture_observation, remaining)
        captured = fresh_camera_images(observation, profile, previous)
        if captured is not None:
            images, stamps = captured
            state = validate_state(operator.get_qpos(observation))
            ensure_robot_healthy(operator)
            previous.update(stamps)
            return {"observation/state": state, "prompt": profile["policy"]["prompt"], **images}, stamps
        time.sleep(min(0.005, max(0, deadline - time.monotonic())))


def run_episode(operator, connection, profile, *, max_steps, chunk_size, episode, previous_stamps):
    step = 0
    while step < max_steps:
        observation, stamps = capture_policy_observation(operator, profile, previous_stamps)
        started = time.monotonic()
        actions = connection.infer(observation)
        count = min(chunk_size, max_steps - step)
        commands = prepare_actions(actions[:count], episode=episode, step=step)
        previous_action = ensure_robot_healthy(operator).astype(np.float64)
        # Validate every interpolated point before the first send of this chunk.
        trajectories = []
        for offset, command in enumerate(commands):
            trajectory = interpolate_action(previous_action, command)
            trajectories.append(prepare_actions(trajectory, episode=episode, step=step + offset))
            previous_action = command
        emit("chunk_ready", episode=episode, policy_step=step, targets=count,
             inference_s=time.monotonic() - started, camera_timestamps_ns=stamps,
             raw_actions=actions[:count].tolist(), command_actions=commands.tolist())
        completed = 0
        for offset, trajectory in enumerate(trajectories):
            age = (time.time_ns() - min(stamps.values())) / 1e9
            if age > profile["robot"]["max_camera_age_s"]:
                raise TimeoutError("Observation aged past the action deadline; stopping the current chunk")
            remaining = profile["robot"]["max_camera_age_s"] - age
            required = len(trajectory) / profile["policy"]["fps"]
            # Keep the existing waypoint cadence and finish only whole targets
            # that fit. Replan from fresh images/state rather than starting an
            # old target that predictably expires partway through execution.
            if required > remaining:
                if completed == 0:
                    raise TimeoutError(
                        f"First action target needs {required:.3f}s but the observation has only "
                        f"{remaining:.3f}s remaining; refusing motion without a feasible budget"
                    )
                emit("chunk_replan", episode=episode, policy_step=step + completed,
                     completed_targets=completed, discarded_targets=count - completed,
                     observation_age_s=age, remaining_s=remaining, required_s=required,
                     reason="next_target_exceeds_observation_budget")
                break
            for waypoint in trajectory:
                ensure_robot_healthy(operator)
                if (time.time_ns() - min(stamps.values())) / 1e9 > profile["robot"]["max_camera_age_s"]:
                    raise TimeoutError("Observation aged past the action deadline; stopping the current chunk")
                sent = time.monotonic()
                operator.send_action(waypoint)
                time.sleep(max(0.0, sent + 1.0 / profile["policy"]["fps"] - time.monotonic()))
            # A stall on the final waypoint must not be hidden by replanning
            # or by reaching max_steps without another pre-send check.
            if (time.time_ns() - min(stamps.values())) / 1e9 > profile["robot"]["max_camera_age_s"]:
                raise TimeoutError("Observation aged past the action deadline; stopping the current chunk")
            emit("policy_step_sent", episode=episode, policy_step=step + offset,
                 waypoints=len(trajectory), command=commands[offset].tolist())
            completed += 1
        step += completed


def robot_session(args, profile, *, hardware_loader=load_hardware, connection_factory=PolicyConnection, input_fn=input):
    reset = prepare_actions(np.asarray(args.reset_action)[None], reset=True)[0]
    if not 1 <= args.chunk_size_execute <= profile["policy"]["horizon"] or args.max_steps < 1:
        raise ValueError("max-steps must be positive; chunk-size-execute must be in [1,8]")
    if not args.can_interface or not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.can_interface):
        raise ValueError("An explicit valid CAN interface is required")
    if not args.execute:
        print(json.dumps({"execute": False, "hardware_imported": False, "profile_id": profile["profile_id"],
                          "checkpoint_sha256": profile["policy"]["checkpoint_sha256"],
                          "host": args.host, "port": args.port, "can_interface": args.can_interface,
                          "reset_action": reset.tolist(), "max_steps": args.max_steps,
                          "chunk_size_execute": args.chunk_size_execute, "fps": profile["policy"]["fps"],
                          "control_frequency": profile["robot"]["control_frequency"]},
                         ensure_ascii=False, indent=2), flush=True)
        return
    with connection_factory(profile, args.host, args.port) as connection:
        # No SDK import or RobotAH construction precedes validated server identity.
        bindings = hardware_loader(args.openpi_root)
        settings = profile["robot"]
        config = bindings.RobotAHConfig(robot_type="play", robot_groups=["left"], robot_interface=[args.can_interface],
                                        control_frequency=settings["control_frequency"],
                                        **{key: settings[key] for key in ("camera_names", "camera_index", "camera_widths",
                                                                          "camera_heights", "camera_fps")})
        if getattr(config, "control_frequency", None) != settings["control_frequency"]:
            raise ValueError("Hardware configuration did not preserve the verified control frequency")
        operator = None
        try:
            operator = bindings.RobotAH(config)
            ensure_robot_healthy(operator)
            emit("hardware_initialized_holding", checkpoint_sha256=profile["policy"]["checkpoint_sha256"],
                 can_interface=args.can_interface, control_frequency=settings["control_frequency"],
                 next_action="Enter explicitly starts reset and run")
            episode, timestamps = 0, {}
            while True:
                answer = input_fn("Enter: reset and run; q + Enter: disconnect without resetting: ")
                if answer.strip().lower() == "q":
                    emit("quit_without_reset", episode=episode)
                    return
                if answer != "":
                    continue
                ensure_robot_healthy(operator)
                episode += 1
                emit("reset_requested", episode=episode, reset_action=reset.tolist())
                operator.switch_mode(bindings.SystemMode.RESETTING)
                operator.set_speed_profile(bindings.SpeedProfile.SLOW)
                operator.move_to_joint_pos(reset.copy())
                ensure_robot_healthy(operator)
                operator.switch_mode(bindings.SystemMode.SAMPLING)
                operator.set_speed_profile(bindings.SpeedProfile.FAST)
                run_episode(operator, connection, profile, max_steps=args.max_steps,
                            chunk_size=args.chunk_size_execute, episode=episode, previous_stamps=timestamps)
                operator.set_speed_profile(bindings.SpeedProfile.SLOW)
                emit("episode_finished_holding", episode=episode, policy_steps=args.max_steps)
        finally:
            if operator is not None:
                emit("hardware_shutdown", automatic_reset=False)
                if operator.shutdown() is False:
                    LOGGER.error("Hardware shutdown reported failure; physical motor state must be checked locally")


def camera_check(args, profile):
    from PIL import Image
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    factory = _upstream_module(args.openpi_root, "camera_sources").create_camera
    _upstream_module(args.openpi_root, "rtsp_camera")
    settings, cameras, previous, records = profile["robot"], {}, {}, []
    try:
        for name, source, width, height, fps in zip(settings["camera_names"], settings["camera_index"],
                                                    settings["camera_widths"], settings["camera_heights"], settings["camera_fps"], strict=True):
            camera = factory(source, width, height, fps)
            cameras[name] = camera
            if not camera.configure():
                raise RuntimeError(f"Camera failed to configure: {name}")
        for number in range(5):
            deadline = time.monotonic() + settings["max_camera_age_s"]
            while True:
                def capture():
                    return {f"{name}/{key}": value for name, camera in cameras.items()
                            for key, value in camera.capture_observation().items()}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Camera timestamps did not advance")
                observation = capture_before_deadline(capture, remaining)
                result = fresh_camera_images(observation, profile, previous)
                if result is not None:
                    images, stamps = result
                    previous.update(stamps)
                    records.append({"frame": number, "camera_timestamps_ns": stamps})
                    emit("cameras_updated", **records[-1])
                    break
                time.sleep(0.005)
        output.mkdir(parents=True, exist_ok=False)
        for name in settings["camera_names"]:
            Image.fromarray(images[f"observation/{name}"]).save(output / f"{name}.png")
        write_new_json(output / "camera_report.json", {"robot_constructed": False, "frames": records,
                                                       "input_color": "RGB", "resized": False,
                                                       "profile_id": profile["profile_id"]})
    finally:
        for name, camera in cameras.items():
            try:
                if camera.shutdown() is False:
                    LOGGER.error("Camera shutdown reported failure: %s", name)
            except Exception:
                LOGGER.exception("Camera shutdown failed: %s", name)


def compare_reference(observation_path, actions, checkpoint_sha):
    reference_path = Path(observation_path).parent / "reference_actions.json"
    if not reference_path.is_file():
        return {"status": "not_available"}
    reference = json.loads(reference_path.read_text())
    identity = {"path": str(reference_path.resolve()), "sha256": sha256_file(reference_path)}
    if reference.get("sample_observation_sha256") != sha256_file(observation_path):
        return {**identity, "status": "skipped", "reason": "Reference is not bound to this exact observation SHA256"}
    if reference.get("checkpoint_sha256", reference.get("exported_checkpoint_sha256")) != checkpoint_sha:
        raise ValueError("Reference actions belong to a different checkpoint")
    expected = validate_actions(reference.get("actions", reference.get("bundle_actions")))
    difference = float(np.max(np.abs(actions - expected)))
    if not np.allclose(actions, expected, atol=1e-6, rtol=0):
        raise ValueError(f"Probe differs from reference actions: max_abs_difference={difference}")
    return {**identity, "status": "passed", "absolute_tolerance": 1e-6, "max_abs_difference": difference}


def probe(args, profile):
    if Path(args.report).exists():
        raise FileExistsError(args.report)
    with np.load(args.observation, allow_pickle=False) as sample:
        if set(sample.files) != {"state", *CAMERA_KEYS}:
            raise ValueError("Probe NPZ must contain state and exactly the native RGB camera keys")
        observation = {"observation/state": validate_state(sample["state"]), "prompt": profile["policy"]["prompt"]}
        for index, (wire, native) in enumerate(CAMERA_MAPPING.items()):
            observation[wire] = validate_rgb(sample[native], profile["robot"]["camera_widths"][index],
                                             profile["robot"]["camera_heights"][index])
    with PolicyConnection(profile, args.host, args.port) as connection:
        started = time.monotonic()
        actions = connection.infer(observation)
        elapsed = time.monotonic() - started
        result = {"schema_version": 1, "robot_executed": False, "hardware_imported": False,
                  "checkpoint_sha256": connection.metadata["checkpoint_sha256"],
                  "sample_observation_sha256": sha256_file(args.observation), "metadata": connection.metadata,
                  "actions": actions.tolist(), "request_seconds": elapsed,
                  "joint_limit_violations": np.argwhere((actions[:, :6] < JOINT_LIMITS[:, 0]) |
                                                         (actions[:, :6] > JOINT_LIMITS[:, 1])).tolist(),
                  "gripper_out_of_range_rows": np.flatnonzero((actions[:, 6] < GRIPPER_LIMITS[0]) |
                                                                 (actions[:, 6] > GRIPPER_LIMITS[1])).tolist()}
        result["reference_comparison"] = compare_reference(args.observation, actions, result["checkpoint_sha256"])
    write_new_json(args.report, result)
    emit("probe_complete", report=str(Path(args.report).resolve()), request_seconds=elapsed,
         checkpoint_sha256=result["checkpoint_sha256"], robot_executed=False)


class SignalStop(Exception):
    def __init__(self, signum):
        self.signum = signum


@contextmanager
def signal_cleanup():
    def stop(signum, frame):
        raise SignalStop(signum)
    previous = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "cameras", "robot"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--profile", required=True)
        subparser.add_argument("--host", default="127.0.0.1")
        subparser.add_argument("--port", type=int, default=8026)
        subparser.add_argument("--openpi-root", required=True)
        if name == "probe":
            subparser.add_argument("--observation", required=True)
            subparser.add_argument("--report", required=True)
        elif name == "cameras":
            subparser.add_argument("--output-dir", required=True)
        else:
            subparser.add_argument("--can-interface", required=True)
            subparser.add_argument("--reset-action", nargs=7, type=float, required=True)
            subparser.add_argument("--max-steps", type=int, default=25)
            subparser.add_argument("--chunk-size-execute", type=int, default=4)
            subparser.add_argument("--execute", action="store_true")
            subparser.add_argument("--log")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    handlers = [logging.StreamHandler(sys.stdout)]
    if getattr(args, "log", None):
        path = Path(args.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="x", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    try:
        profile = load_profile(args.profile)
        with signal_cleanup():
            if args.command == "probe":
                probe(args, profile)
            elif args.command == "cameras":
                camera_check(args, profile)
            else:
                robot_session(args, profile)
        return 0
    except (SignalStop, KeyboardInterrupt) as error:
        number = error.signum if isinstance(error, SignalStop) else signal.SIGINT
        emit("interrupted", signal=number, automatic_reset=False)
        return 128 + number
    except Exception as error:
        LOGGER.error("Stopped without replaying actions: %s: %s", type(error).__name__, error)
        return 1
    finally:
        for handler in handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
