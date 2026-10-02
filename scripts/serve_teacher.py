import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", default="/home/user/wang-sm/openpi")
    parser.add_argument("--checkpoint", default="/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--platform", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.platform == "cuda" and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        raise RuntimeError("Set an explicit XLA memory fraction after checking GPU availability")
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.platform == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(args.openpi_root) / "src"))
    import jax
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    expected_backend = "cpu" if args.platform == "cpu" else "gpu"
    if any(device.platform != expected_backend for device in jax.devices()):
        raise RuntimeError("Unexpected inference backend")
    policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), args.checkpoint)
    policy._rng = jax.random.key(args.seed)
    class SeededTeacher:
        def infer(self, observation):
            observation = dict(observation)
            request_seed = observation.pop("depth_policy_request_seed")
            policy._rng = jax.random.key(int(request_seed))
            return policy.infer(observation)
    metadata = {**policy.metadata, "policy_config": "pi05_libero",
                "checkpoint_path": str(Path(args.checkpoint).resolve()), "teacher_rng_seed": args.seed,
                "request_seed_protocol": 1, "execution_backend": args.platform}
    print(json.dumps({"server_ready": True, "port": args.port, **metadata}), flush=True)
    WebsocketPolicyServer(SeededTeacher(), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
