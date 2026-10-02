"""Serve RGB-to-depth Airbot policy inference over the OpenPI websocket protocol.

This process only returns action arrays. It has no robot SDK or hardware
connection. The default unauthenticated listener is localhost; remote clients
can reach it through a separately configured SSH tunnel. /healthz is liveness,
not evidence that a robot or its cameras have been connected.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
import http
import logging
import os
import re
import time

import numpy as np
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import serve


logger = logging.getLogger(__name__)
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
WIRE_CAMERA_KEYS = {
    "observation/base_0_rgb": "observation.images.head",
    "observation/left_wrist_0_rgb": "observation.images.wrist",
}


class AirbotWebsocketAdapter:
    """Map the known Airbot wire camera names into the checkpoint's camera order."""

    def __init__(self, policy):
        self._policy = policy
        self._camera_keys = list(policy.camera_keys)
        if len(self._camera_keys) != 2 or set(self._camera_keys) != set(WIRE_CAMERA_KEYS.values()):
            raise ValueError("This adapter requires exactly observation.images.head and observation.images.wrist")
        self._native_to_wire = {native: wire for wire, native in WIRE_CAMERA_KEYS.items()}
        manifest = policy.manifest
        if (manifest.get("state_dim") != 7 or manifest.get("action_dim") != 7
                or manifest.get("action_semantics") != "absolute_joint_position"):
            raise ValueError("Airbot websocket serving requires seven-dimensional absolute joint positions")
        for kind in ("state", "action"):
            names = manifest.get(f"{kind}_names")
            if (not isinstance(names, list) or len(names) != 7
                    or any(not isinstance(name, str) or not name for name in names)):
                raise ValueError(f"Checkpoint must record all seven {kind} names")
            if manifest.get(f"{kind}_units") != ["rad"] * 6 + ["m"]:
                raise ValueError(f"Checkpoint must record six joint radians and gripper metres for {kind}")
        if not isinstance(policy.prompt, str) or not policy.prompt.strip():
            raise ValueError("Checkpoint must contain its exact training prompt")
        if not np.isfinite(policy.fps) or policy.fps <= 0:
            raise ValueError("Checkpoint frame rate must be finite and positive")
        horizon = policy.model.horizon
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Checkpoint action horizon must be a positive integer")
        if re.fullmatch(r"[0-9a-f]{64}", policy.checkpoint_sha256) is None:
            raise ValueError("Checkpoint must have a SHA256 identity")
        if not manifest.get("depth_config") or not manifest.get("depth_provenance"):
            raise ValueError("Checkpoint must contain its frozen depth preprocessing provenance")
        self._horizon = horizon
        self._prompt = policy.prompt
        self._metadata = {
            "schema_version": 1,
            "policy_type": "airbot_depth",
            "protocol": "openpi_msgpack_numpy",
            "fps": float(policy.fps),
            "horizon": horizon,
            "state_dim": 7,
            "action_dim": 7,
            "state_names": deepcopy(manifest["state_names"]),
            "action_names": deepcopy(manifest["action_names"]),
            "state_units": deepcopy(manifest["state_units"]),
            "action_units": deepcopy(manifest["action_units"]),
            "action_semantics": "absolute_joint_position",
            "prompt": self._prompt,
            "camera_keys": list(self._camera_keys),
            "camera_mapping": dict(WIRE_CAMERA_KEYS),
            "input_color": "RGB",
            "preprocessing": {
                "depth_config": deepcopy(manifest["depth_config"]),
                "depth_provenance": deepcopy(manifest["depth_provenance"]),
                "model_input_quantization": manifest.get("model_input_quantization"),
            },
            "checkpoint_sha256": policy.checkpoint_sha256,
            "max_message_bytes": MAX_MESSAGE_BYTES,
        }

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def infer(self, observation):
        expected = {"observation/state", "prompt", *WIRE_CAMERA_KEYS}
        if not isinstance(observation, Mapping) or set(observation) != expected:
            raise ValueError(f"Observation must contain exactly these keys: {sorted(expected)}")
        if not isinstance(observation["prompt"], str) or observation["prompt"] != self._prompt:
            raise ValueError("The request prompt must exactly match this checkpoint's training prompt")
        state = np.asarray(observation["observation/state"])
        if state.shape != (7,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
            raise ValueError("observation/state must contain seven finite joint/gripper values")
        with np.errstate(over="ignore"):
            state = state.astype(np.float32, copy=False)
        if not np.isfinite(state).all():
            raise ValueError("observation/state cannot be represented as finite float32 values")
        images = {}
        for native in self._camera_keys:
            wire = self._native_to_wire[native]
            image = observation[wire]
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2):
                raise ValueError(f"{wire} must be a uint8 HWC RGB image with three channels")
            images[native] = image
        actions = np.asarray(self._policy.infer_rgb(state, images, prompt=self._prompt))
        if (actions.shape != (self._horizon, 7) or actions.dtype.kind not in "fiu"
                or not np.isfinite(actions).all()):
            raise FloatingPointError("Policy returned invalid absolute action shape or nonfinite values")
        with np.errstate(over="ignore"):
            # Snapshot the result before another request can reuse policy buffers.
            actions = np.array(actions, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(actions).all():
            raise FloatingPointError("Policy actions cannot be represented as finite float32 values")
        return {"actions": actions}


async def _health_check(path, request_headers):
    if path == "/healthz":
        return http.HTTPStatus.OK, [("Content-Type", "text/plain; charset=utf-8")], b"OK\n"
    return None


class AirbotWebsocketServer:
    """Bounded websocket transport with one shared inference worker for all clients."""

    def __init__(self, adapter, host="127.0.0.1", port=8026):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer from 0 to 65535")
        self.adapter = adapter
        self.host = host
        self.port = port
        self._executor = None

    @asynccontextmanager
    async def listen(self):
        """Yield a bound server; also supports port 0 for a temporary local test."""
        if self._executor is not None:
            raise RuntimeError("This server instance is already listening")
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="airbot-depth-inference") as executor:
            self._executor = executor
            try:
                async with serve(
                    self._handler, self.host, self.port,
                    compression=None, max_size=MAX_MESSAGE_BYTES, max_queue=1,
                    process_request=_health_check, close_timeout=5,
                ) as server:
                    yield server
            finally:
                self._executor = None

    async def run(self):
        async with self.listen() as server:
            addresses = [sock.getsockname() for sock in server.sockets]
            logger.info("Airbot depth policy listening on %s; checkpoint=%s", addresses,
                        self.adapter.metadata["checkpoint_sha256"])
            await server.serve_forever()

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        try:
            await websocket.send(packer.pack(self.adapter.metadata))
            while True:
                payload = await websocket.recv()
                if not isinstance(payload, bytes):
                    raise ValueError("Inference requests must be binary msgpack messages")
                observation = msgpack_numpy.unpackb(payload)
                started = time.monotonic()
                response = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self.adapter.infer, observation
                )
                response["server_timing"] = {"infer_ms": (time.monotonic() - started) * 1000}
                await websocket.send(packer.pack(response))
        except ConnectionClosed:
            return
        except Exception as error:
            logger.warning("Inference connection rejected: %s: %s", type(error).__name__, error)
            try:
                await websocket.send(f"{type(error).__name__}: {error}"[:2000])
                await websocket.close(code=1011, reason="Inference failed; see preceding text message")
            except ConnectionClosed:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted local Airbot depth policy checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8026)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4, help="Torch CPU threads; inference remains serial")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    from airbot_depth.policy import AirbotDepthPolicy

    if str(args.device).startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = AirbotDepthPolicy(args.checkpoint, device=args.device, local_files_only=args.local_files_only)
    server = AirbotWebsocketServer(AirbotWebsocketAdapter(policy), host=args.host, port=args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Airbot depth inference server stopped")


if __name__ == "__main__":
    main()
