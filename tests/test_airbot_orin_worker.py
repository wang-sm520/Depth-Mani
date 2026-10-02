"""Opt-in real-GPU regression at the same thread boundary used by serving."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("AIRBOT_GPU_WORKER_TEST") != "1", reason="requires the audited local CUDA bundle")
def test_actual_policy_has_same_numerics_in_the_server_worker():
    from airbot_depth.depth import DepthAnythingTransform
    from airbot_depth.policy import AirbotDepthPolicy
    from airbot_orin.runtime import AttestedPolicy, configure_runtime

    root = Path(__file__).resolve().parents[1]
    bundle = root / "deploy/airbot-paperbag200-da2-100k-20260924"
    os.environ["HF_HUB_CACHE"] = str(bundle / "hf_hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    configure_runtime("cuda", 4)
    policy = AirbotDepthPolicy(bundle / "policy.pt", device="cuda", local_files_only=True)
    transform = DepthAnythingTransform(policy.depth_config, device="cuda", local_files_only=True)
    wrapped = AttestedPolicy(policy, transform)
    # The original bundle's export comparison used default cuDNN TF32. Native
    # validation and serving explicitly preserve the training audit's TF32=False.
    fixture = root / "deploy/airbot-paperbag200-da2-100k-orin-reference-20260924/samples/episode_000005_frame_000000.npz"
    with np.load(fixture, allow_pickle=False) as sample:
        state = sample["state"]
        images = {key: sample[key] for key in policy.camera_keys}
        expected = sample["actions_rgb"]
    main = wrapped.infer_rgb(state, images, prompt=policy.prompt)
    np.testing.assert_array_equal(main, expected)
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(wrapped.infer_rgb, state, images, prompt=policy.prompt).result(timeout=15)
    np.testing.assert_array_equal(worker, main)
