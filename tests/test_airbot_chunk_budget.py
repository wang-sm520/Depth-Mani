"""Replay the reported Orin chunk using virtual time and no hardware imports."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airbot_deploy import client


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = np.array([-0.08783547580242157, -1.0316509008407593, 0.9035974740982056,
                     -0.5324294567108154, 0.5848516225814819, 0.061737217009067535,
                     0.07164420932531357])
TARGETS = np.array([
    [-0.0872897282242775, -1.048844814300537, 0.9960584044456482, -0.7900063395500183,
     0.7195542454719543, 0.27562347054481506, 0.07103417813777924],
    [-0.08695841580629349, -1.0533461570739746, 0.9984142780303955, -0.7955193519592285,
     0.7108069062232971, 0.2800554037094116, 0.07039060443639755],
    [-0.08723340183496475, -1.0651648044586182, 1.0063914060592651, -0.8027622103691101,
     0.6942687034606934, 0.28073248267173767, 0.07020606100559235],
    [-0.0860552191734314, -1.0880323648452759, 1.0215858221054077, -0.8072052597999573,
     0.6843008399009705, 0.2729054093360901, 0.06976310908794403],
])


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time_ns(self):
        return round((100000.0 + self.now) * 1e9)

    def sleep(self, duration):
        self.now += duration


class Operator:
    def __init__(self, clock, profile):
        self.clock = clock
        self.position = PREVIOUS.copy()
        self.sent = []
        self.capture_count = 0
        self.freeze_after_first = False
        self.first_stamp = None
        self.stall_once = 0.0
        settings = profile["robot"]
        self.images = {name: np.zeros((height, width, 3), dtype=np.uint8)
                       for name, width, height in zip(settings["camera_names"], settings["camera_widths"],
                                                      settings["camera_heights"], strict=True)}
        self.robots = {"left": SimpleNamespace(_is_connected=True, _control_error=None,
                      arm=SimpleNamespace(state=lambda: SimpleNamespace(is_valid=True, pos=self.position.copy())))}

    def capture_observation(self):
        self.capture_count += 1
        stamp = self.clock.time_ns() - 50_000_000
        self.first_stamp = stamp if self.first_stamp is None else self.first_stamp
        if self.freeze_after_first:
            stamp = self.first_stamp
        return {"position": self.position.copy(), **{
            f"{name}/color/image_raw": {"data": image, "t": stamp} for name, image in self.images.items()}}

    def get_qpos(self, observation):
        return observation["position"]

    def send_action(self, action):
        self.sent.append((self.clock.now, action.copy()))
        self.position = action.copy()
        if self.stall_once:
            self.clock.sleep(self.stall_once)
            self.stall_once = 0.0


class Policy:
    def __init__(self, clock):
        self.clock = clock
        self.requests = []
        self.first = np.tile(TARGETS, (2, 1))
        self.delay = 0.751609938

    def infer(self, observation):
        self.requests.append(deepcopy(observation["observation/state"]))
        self.clock.sleep(self.delay)
        if len(self.requests) == 1:
            return self.first.copy()
        action = observation["observation/state"].copy()
        action[0] += 0.001
        return np.tile(action, (8, 1))


@pytest.fixture
def replay(monkeypatch):
    profile = client.load_profile(ROOT / "configs/airbot_paperbag_deploy.json")
    clock = Clock()
    operator = Operator(clock, profile)
    policy = Policy(clock)
    events = []
    monkeypatch.setattr(client, "time", clock)
    monkeypatch.setattr(client, "emit", lambda event, **fields: events.append({"event": event, **fields}))
    return profile, clock, operator, policy, events


def run(replay, max_steps=4):
    profile, clock, operator, policy, events = replay
    client.run_episode(operator, policy, profile, max_steps=max_steps, chunk_size=4,
                       episode=1, previous_stamps={})


def test_reported_chunk_refreshes_before_a_target_cannot_fit(replay):
    profile, clock, operator, policy, events = replay
    run(replay)
    assert len(policy.requests) == 2
    np.testing.assert_array_equal(policy.requests[1], TARGETS[2])
    sent_steps = [event for event in events if event["event"] == "policy_step_sent"]
    assert [event["policy_step"] for event in sent_steps] == [0, 1, 2, 3]
    assert [event["waypoints"] for event in sent_steps[:3]] == [26, 1, 2]
    assert not any(np.array_equal(action, TARGETS[3]) for _, action in operator.sent)
    assert profile["robot"]["max_camera_age_s"] == 2
    assert all(b[0] - a[0] >= .04 - 1e-10 for a, b in zip(operator.sent, operator.sent[1:]))


def test_first_target_that_cannot_fit_stops_without_retry_or_motion(replay):
    profile, clock, operator, policy, events = replay
    policy.first[:, 3] = -1.3
    with pytest.raises(TimeoutError):
        run(replay)
    assert len(policy.requests) == 1 and not operator.sent


def test_refresh_requires_new_camera_frames_and_does_not_send_old_remainder(replay):
    profile, clock, operator, policy, events = replay
    operator.freeze_after_first = True
    with pytest.raises(TimeoutError, match="camera|Camera"):
        run(replay)
    assert len(policy.requests) == 1
    assert len(operator.sent) == 29
    np.testing.assert_array_equal(operator.position, TARGETS[2])


def test_unexpected_stall_during_a_target_still_stops_immediately(replay):
    profile, clock, operator, policy, events = replay
    operator.stall_once = 2.1
    with pytest.raises(TimeoutError, match="aged past"):
        run(replay)
    assert len(operator.sent) == 1 and len(policy.requests) == 1


@pytest.mark.parametrize("max_steps", [1, 4])
def test_stall_on_a_targets_last_waypoint_stops_even_at_episode_end(replay, max_steps):
    profile, clock, operator, policy, events = replay
    policy.first = np.tile(PREVIOUS, (8, 1))
    operator.stall_once = 2.1
    with pytest.raises(TimeoutError, match="aged past"):
        run(replay, max_steps=max_steps)
    assert len(operator.sent) == 1 and len(policy.requests) == 1
    assert not any(event["event"] == "chunk_replan" for event in events)


def test_short_chunks_keep_the_original_execution_count(replay):
    profile, clock, operator, policy, events = replay
    policy.first = np.tile(PREVIOUS, (8, 1))
    run(replay, max_steps=5)
    assert len(policy.requests) == 2 and len(operator.sent) == 5
    assert [event["policy_step"] for event in events if event["event"] == "policy_step_sent"] == list(range(5))
