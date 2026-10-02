import asyncio
from copy import deepcopy
from types import SimpleNamespace
import threading
import time
import urllib.request

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.legacy.client import connect

from airbot_depth.serve import AirbotWebsocketAdapter, AirbotWebsocketServer, MAX_MESSAGE_BYTES


class FakePolicy:
    def __init__(self):
        # Deliberately reverse the conventional ordering; the adapter must use
        # the checkpoint's declared camera order and preserve each RGB array.
        self.camera_keys = ["observation.images.wrist", "observation.images.head"]
        self.prompt = "pick up the red paper bag and hold in the reset position"
        self.fps = 25.0
        self.model = SimpleNamespace(horizon=4)
        self.checkpoint_sha256 = "a" * 64
        self.manifest = {
            "state_dim": 7, "action_dim": 7,
            "state_names": [f"joint_{index}" for index in range(6)] + ["gripper"],
            "action_names": [f"joint_{index}" for index in range(6)] + ["gripper"],
            "state_units": ["rad"] * 6 + ["m"],
            "action_units": ["rad"] * 6 + ["m"],
            "action_semantics": "absolute_joint_position",
            "depth_config": {"image_size": 128, "representation": "relative_inverse_depth"},
            "depth_provenance": {"fixture": "without_weights"},
            "model_input_quantization": "float32_to_float16_to_float32",
        }
        self.calls = []
        self.active_calls = 0
        self.maximum_active_calls = 0
        self._lock = threading.Lock()

    def infer_rgb(self, state, images, prompt=None):
        with self._lock:
            self.active_calls += 1
            self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            time.sleep(0.02)
            self.calls.append((state.copy(), deepcopy(images), prompt))
            return np.tile(state, (self.model.horizon, 1))
        finally:
            with self._lock:
                self.active_calls -= 1


def observation(policy, first_state=0.0):
    state = np.arange(7, dtype=np.float32)
    state[0] = first_state
    return {
        "observation/state": state,
        "observation/base_0_rgb": np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3),
        "observation/left_wrist_0_rgb": np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3),
        "prompt": policy.prompt,
    }


def test_existing_openpi_client_exchanges_metadata_and_actions_with_local_server():
    policy = FakePolicy()
    adapter = AirbotWebsocketAdapter(policy)
    request = observation(policy)

    def client_exchange(port):
        client = WebsocketClientPolicy(host="127.0.0.1", port=port,
                                       connect_timeout_s=3, inference_timeout_s=3)
        try:
            metadata = client.get_server_metadata()
            response = client.infer(request)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as health:
                assert health.status == 200
                assert health.read() == b"OK\n"
            return metadata, response
        finally:
            client.close()

    async def exchange():
        async with AirbotWebsocketServer(adapter, port=0).listen() as server:
            return await asyncio.to_thread(client_exchange, server.sockets[0].getsockname()[1])

    metadata, response = asyncio.run(exchange())
    assert metadata["fps"] == 25
    assert metadata["horizon"] == 4
    assert metadata["checkpoint_sha256"] == "a" * 64
    assert metadata["max_message_bytes"] == MAX_MESSAGE_BYTES
    assert metadata["action_units"] == ["rad"] * 6 + ["m"]
    assert metadata["preprocessing"]["depth_config"] == policy.manifest["depth_config"]
    np.testing.assert_array_equal(response["actions"], np.tile(request["observation/state"], (4, 1)))
    assert response["actions"].dtype == np.float32
    assert response["server_timing"]["infer_ms"] > 0
    state, images, prompt = policy.calls[0]
    assert list(images) == policy.camera_keys
    assert prompt == policy.prompt
    np.testing.assert_array_equal(state, request["observation/state"])
    np.testing.assert_array_equal(images["observation.images.head"], request["observation/base_0_rgb"])
    np.testing.assert_array_equal(images["observation.images.wrist"], request["observation/left_wrist_0_rgb"])


@pytest.mark.parametrize("replacement", [
    {"prompt": "different instruction"},
    {"observation/state": np.full(7, np.nan)},
    {"observation/state": np.zeros(6)},
    {"observation/base_0_rgb": np.zeros((8, 12, 3), dtype=np.float32)},
    {"observation/left_wrist_0_rgb": np.zeros((3, 8, 12), dtype=np.uint8)},
    {"extra": True},
])
def test_adapter_rejects_invalid_observation_before_policy_call(replacement):
    policy = FakePolicy()
    with pytest.raises(ValueError):
        AirbotWebsocketAdapter(policy).infer(observation(policy) | replacement)
    assert not policy.calls


def test_bad_request_after_success_receives_text_and_closes_without_stale_action():
    policy = FakePolicy()

    async def exchange():
        async with AirbotWebsocketServer(AirbotWebsocketAdapter(policy), port=0).listen() as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                assert isinstance(await client.recv(), bytes)
                await client.send(msgpack_numpy.packb(observation(policy)))
                assert "actions" in msgpack_numpy.unpackb(await client.recv())
                request = observation(policy) | {"prompt": "different instruction"}
                await client.send(msgpack_numpy.packb(request))
                failure = await client.recv()
                assert isinstance(failure, str)
                assert "prompt" in failure
                with pytest.raises(ConnectionClosed) as error:
                    await client.recv()
                assert error.value.code == 1011

    asyncio.run(exchange())
    assert len(policy.calls) == 1


def test_multiple_clients_share_serial_inference_and_keep_responses_distinct():
    policy = FakePolicy()

    async def exchange():
        async with AirbotWebsocketServer(AirbotWebsocketAdapter(policy), port=0).listen() as server:
            port = server.sockets[0].getsockname()[1]

            async def request(value):
                async with connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                    await client.recv()
                    await client.send(msgpack_numpy.packb(observation(policy, value)))
                    return msgpack_numpy.unpackb(await client.recv())["actions"]

            return await asyncio.gather(request(1.25), request(2.75))

    first, second = asyncio.run(exchange())
    np.testing.assert_array_equal(first[:, 0], 1.25)
    np.testing.assert_array_equal(second[:, 0], 2.75)
    assert policy.maximum_active_calls == 1


def test_adapter_rejects_unknown_camera_or_units_without_guessing():
    policy = FakePolicy()
    policy.camera_keys = ["observation.images.front", "observation.images.wrist"]
    with pytest.raises(ValueError, match="requires exactly"):
        AirbotWebsocketAdapter(policy)
    policy = FakePolicy()
    policy.manifest["action_units"][-1] = "normalized"
    with pytest.raises(ValueError, match="metres"):
        AirbotWebsocketAdapter(policy)
