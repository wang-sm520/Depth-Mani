"""Exercise the native helper boundary using fake cameras, CAN hardware and time."""

from contextlib import contextmanager
from functools import partial
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from airbot_deploy import client


ROOT = Path(__file__).resolve().parents[1]
HARDWARE_ROOT = Path(os.environ.get(
    "AIRBOT_TEST_HARDWARE_ROOT", str(ROOT / "vendor/airbot_orin_20260924/openpi")))
RESET = [0.0] * 6 + [0.07]


class FakeCamera:
    def __init__(self, config):
        self.config = config

    def configure(self):
        return True

    def shutdown(self):
        return True


@pytest.fixture
def helpers(monkeypatch):
    def stub(name, **attributes):
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            if key not in sys.modules:
                module = ModuleType(key)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, key, module)
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("airbot_hardware_py")
    modes = SimpleNamespace(PASSIVE="passive", RESETTING="resetting", SAMPLING="sampling")
    stub("airbot_data_collection.basis", SystemMode=modes)
    stub("airbot_data_collection.airbot.sensors.cameras.realsense", RealSense=FakeCamera)
    stub("airbot_data_collection.airbot.sensors.cameras.v4l2", BsonV4L2Camera=FakeCamera,
         V4L2CameraConfig=SimpleNamespace)
    stub("airbot_data_collection.common.devices.cameras.intelrealsense",
         IntelRealSenseCameraConfig=SimpleNamespace)
    stub("rtsp_camera", RTSPCamera=FakeCamera, RTSPCameraConfig=SimpleNamespace)
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name:
                        "0.2.9.2" if name == "airbot-hardware-py" else original_version(name))
    modules = {}
    for name in ("airbot_arm", "robot_config", "camera_sources", "play_operator_ah"):
        path = HARDWARE_ROOT / "examples/airbot" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


def test_profile_frequency_reaches_real_config_and_arm_factory(helpers):
    profile = client.load_profile(ROOT / "configs/airbot_paperbag_deploy.json")
    constructed = []

    class FakeArm:
        def __init__(self, *, interface, frequency=250):
            self.interface, self.frequency = interface, frequency
            self._is_connected, self._control_error = False, None
            self.arm = SimpleNamespace(state=lambda: SimpleNamespace(is_valid=True, pos=RESET))
            constructed.append(self)

        def connect(self):
            self._is_connected = True
            return True

        def disconnect(self):
            self._is_connected = False
            return True

    operator = helpers.play_operator_ah
    bindings = SimpleNamespace(RobotAH=partial(operator.RobotAH, arm_factory=FakeArm),
                               RobotAHConfig=helpers.robot_config.RobotAHConfig,
                               SystemMode=operator.SystemMode, SpeedProfile=operator.SpeedProfile)

    @contextmanager
    def recorded_connection(*args):
        yield SimpleNamespace()

    args = SimpleNamespace(host="unused", port=8026, openpi_root=HARDWARE_ROOT,
                           can_interface="fake-can", reset_action=RESET, max_steps=1,
                           chunk_size_execute=4, execute=True)
    client.robot_session(args, profile, hardware_loader=lambda root: bindings,
                         connection_factory=recorded_connection, input_fn=lambda prompt: "q")
    assert [arm.frequency for arm in constructed] == [100]
    assert all(not arm._is_connected for arm in constructed)


def test_delayed_hardware_send_waits_a_full_period_before_next_send(helpers, monkeypatch):
    module = helpers.airbot_arm
    clock = SimpleNamespace(now=0.0)
    sent_at, pauses = [], []

    def sleep(duration):
        pauses.append(duration)
        clock.now += duration

    class FakeSDKArm:
        def init(self, *args):
            return True

        def enable(self):
            return True

        def set_param(self, *args):
            pass

        def state(self):
            return SimpleNamespace(is_valid=True, pos=RESET)

        def pvt(self, **kwargs):
            sent_at.append(clock.now)
            if len(sent_at) == 3:
                raise RuntimeError("Synthetic end of finite CAN-send fixture")
            clock.now += [0.035, 0.025][len(sent_at) - 1]

        def disable(self):
            return True

        def uninit(self):
            return True

    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, **kwargs):
            pass

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=ImmediateThread, current_thread=lambda: None))
    monkeypatch.setattr(module.ah, "MotorControlMode", SimpleNamespace(PVT="pvt"), raising=False)
    arm = module.AIRBOTArm("fake-can", motor_types=[object()] * 8, frequency=100,
                          arm_factory=lambda *motors: FakeSDKArm(),
                          executor_factory=lambda threads: SimpleNamespace(get_io_context=lambda: None))
    try:
        arm.connect()
    finally:
        arm.disconnect()
    assert sent_at == pytest.approx([0.0, 0.045, 0.080])
    assert pauses == pytest.approx([0.01, 0.01])
