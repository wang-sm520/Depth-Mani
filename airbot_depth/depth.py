"""One frozen relative-depth transform for dataset conversion and RGB inference.

Depth Anything V2 Small predicts relative inverse depth (larger means nearer),
not metres. Each view is normalized separately, then letterboxed. The second
output channel distinguishes far pixels with value zero from invalid padding.
No pseudo-colour rendering is used anywhere in the policy input path.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_DEFAULT_CONFIG = {
    "schema_version": 1,
    "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
    "revision": "5426e4f0f36572d16453bbda7a8389317b1bef99",
    "representation": "relative_inverse_depth",
    "input_color": "RGB",
    "input_dtype": "uint8",
    "input_size": 518,
    "image_size": 128,
    "percentiles": [2.0, 98.0],
    "percentile_method": "linear",
    "normalization_scope": "per_view_per_frame",
    "processor_type": "DPTImageProcessor",
    "processor_use_fast": False,
    "input_resize": "hf_dpt_minimal_aspect_multiple_14_bicubic",
    "processing_order": "percentile_normalize_then_letterbox",
    "output_resize": "bilinear_align_corners_false_antialias_true",
    "output_aspect_ratio": "original_rgb",
    "padding": "center_zero_bottom_right_get_extra_pixel",
    "invalid_prediction": "reject_nonfinite",
    "output_channels": ["relative_inverse_depth", "validity_mask"],
    "output_dtype": "float32",
    "model_dtype": "float32",
    "attention_implementation": "eager",
    "deterministic": True,
    "cublas_workspace_config": ":4096:8",
}

# Public data is safe to serialize or copy into an experiment configuration.
# Validation uses the private template, so modifying this dict cannot weaken it.
DEFAULT_DEPTH_CONFIG = deepcopy(_DEFAULT_CONFIG)
_INFERENCE_LOCK = threading.RLock()


def _validate_percentiles(percentiles: Sequence[float]) -> tuple[float, float]:
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        raise ValueError("percentiles must contain exactly two numbers")
    if any(isinstance(item, (bool, str)) for item in percentiles):
        raise ValueError("percentiles must be finite numbers")
    try:
        low, high = (float(item) for item in percentiles)
    except (TypeError, ValueError) as error:
        raise ValueError("percentiles must be finite numbers") from error
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    return low, high


def _validate_size(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{name} must be an integer of at least 2")
    return value


def canonical_depth_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate supplied overrides and return the complete serializable config.

    Output/input sizes, percentiles and the immutable model commit may be
    selected explicitly. Other choices define this implementation's contract;
    changing them requires a new implementation rather than ignored options.
    """
    if not isinstance(config, dict):
        raise TypeError("Depth Anything config must be a dict")
    unknown = set(config) - set(_DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown Depth Anything config fields: {sorted(unknown)}")
    result = deepcopy(_DEFAULT_CONFIG)
    result.update(deepcopy(config))
    configurable = {"input_size", "image_size", "percentiles", "revision"}
    for name, expected in _DEFAULT_CONFIG.items():
        if name not in configurable and (
            type(result[name]) is not type(expected) or result[name] != expected
        ):
            raise ValueError(f"Unsupported {name}={result[name]!r}; expected {expected!r}")
    revision = result["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be an immutable lowercase 40-character commit SHA")
    result["input_size"] = _validate_size(result["input_size"], "input_size")
    if result["input_size"] % 14:
        raise ValueError("input_size must be a multiple of the model's 14-pixel patch size")
    result["image_size"] = _validate_size(result["image_size"], "image_size")
    result["percentiles"] = list(_validate_percentiles(result["percentiles"]))
    return result


def normalize_relative_depth(
    depth: np.ndarray,
    percentiles: Sequence[float] = (2.0, 98.0),
    image_size: int = 128,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return ``[2, image_size, image_size]`` depth and binary validity mask.

    Percentiles use finite pixels of the original 2-D prediction, with linear
    quantile interpolation. Values are clipped to [0, 1] before bilinear,
    antialiased resizing and centre padding; no depth units are assigned.
    A zero depth remains valid. Non-finite input pixels are excluded and their
    interpolation support is invalidated. Completely invalid or degenerate
    predictions are errors, including when the chosen percentiles coincide.
    ``source_shape`` optionally gives the original RGB height/width: it restores
    that aspect ratio after the model's resize rounded dimensions to patches.

    The model wrapper rejects *any* non-finite prediction. Supporting a partial
    validity mask here also makes this pure postprocessor useful for audits.
    """
    size = _validate_size(image_size, "image_size")
    bounds = _validate_percentiles(percentiles)
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or not depth.size:
        raise ValueError("relative depth must be a nonempty 2-D numpy array")
    if depth.dtype.kind not in "fiu":
        raise ValueError("relative depth must contain real numeric values")
    if source_shape is not None and (
        not isinstance(source_shape, tuple)
        or len(source_shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in source_shape)
    ):
        raise ValueError("source_shape must be a (height, width) tuple of positive integers")
    values = depth.astype(np.float64, copy=False)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("relative depth prediction has no finite valid pixels")
    low, high = np.percentile(values[valid], bounds, method="linear")
    span = high - low
    if not np.isfinite([low, high, span]).all() or span <= 0:
        raise ValueError("relative depth prediction is degenerate at the chosen percentiles")
    normalized = np.zeros(depth.shape, dtype=np.float32)
    # Clip before subtracting to avoid overflow for extreme finite outliers.
    normalized[valid] = ((np.clip(values[valid], low, high) - low) / span).astype(np.float32)

    height, width = depth.shape if source_shape is None else source_shape
    scale = size / max(height, width)
    output_height = max(1, min(size, round(height * scale)))
    output_width = max(1, min(size, round(width * scale)))
    stacked = np.stack([normalized, valid.astype(np.float32)], axis=0)
    resized = F.interpolate(
        torch.from_numpy(stacked).unsqueeze(0),
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0].numpy()
    # Conservative validity: interpolating an invalid source invalidates the
    # destination too. Tolerance only absorbs floating-point summation error.
    resized_valid = resized[1] >= 1.0 - 1e-6
    if not resized_valid.any():
        raise ValueError("relative depth has no valid pixels after resizing")
    resized[0] = np.where(resized_valid, np.clip(resized[0], 0.0, 1.0), 0.0)
    resized[1] = resized_valid.astype(np.float32)
    result = np.zeros((2, size, size), dtype=np.float32)
    top, left = (size - output_height) // 2, (size - output_width) // 2
    result[:, top : top + output_height, left : left + output_width] = resized
    return result


@contextmanager
def _deterministic_float32(device_type: str) -> Iterator[None]:
    """Apply inference settings temporarily, restoring the caller's settings."""
    # Torch's algorithm/TF32 flags are global. Serialize this module's calls,
    # and restore them so a caller's later policy-training setup is unchanged.
    with _INFERENCE_LOCK:
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=False):
                with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                    yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DepthAnythingTransform:
    """Pinned HF Depth Anything V2 Small for offline and online observations.

    Views may have different resolutions; each is processed independently in
    the supplied order. ``__call__`` accepts only uint8 HWC RGB arrays and
    returns float32 ``[view, depth_or_mask, image_size, image_size]``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        # Validate before importing/downloading the large model dependencies.
        self._config = canonical_depth_config(config)
        self._device = torch.device(device)
        if self._device.type not in {"cpu", "cuda"}:
            raise ValueError("Depth Anything inference supports cpu or cuda devices")
        if self._device.type == "cuda":
            workspace = os.environ.setdefault(
                "CUBLAS_WORKSPACE_CONFIG", self._config["cublas_workspace_config"]
            )
            if workspace != self._config["cublas_workspace_config"]:
                raise ValueError("Deterministic depth inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor_settings = {
            "do_resize": True,
            "size": {"height": self._config["input_size"], "width": self._config["input_size"]},
            "resample": 3,  # Pillow BICUBIC; serialize the integer, not an enum.
            "keep_aspect_ratio": True,
            "ensure_multiple_of": 14,
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "do_pad": False,
        }
        load_options = {
            "revision": self._config["revision"],
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        self._processor = AutoImageProcessor.from_pretrained(
            self._config["model_id"], use_fast=False, **load_options, **self._processor_settings
        )
        if type(self._processor).__name__ != self._config["processor_type"]:
            raise ValueError(f"Expected DPTImageProcessor, got {type(self._processor).__name__}")
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self._config["model_id"],
            dtype=torch.float32,
            use_safetensors=True,
            attn_implementation="eager",
            **load_options,
        ).to(device=self._device, dtype=torch.float32).eval()
        if getattr(self._model.config, "depth_estimation_type", None) != "relative":
            raise ValueError("The selected model does not predict relative depth")
        if getattr(self._model.config, "patch_size", None) != 14:
            raise ValueError("The selected model does not use the required 14-pixel patch size")
        self._provenance: dict[str, Any] | None = None

    @property
    def config(self) -> dict[str, Any]:
        """Return a copy so callers cannot mutate an initialized transform."""
        return deepcopy(self._config)

    def __call__(self, rgb_views: list[np.ndarray]) -> np.ndarray:
        if not isinstance(rgb_views, list) or not rgb_views:
            raise ValueError("rgb_views must be a nonempty list of uint8 HWC RGB numpy arrays")
        # Check every view before inference to catch a malformed second camera
        # without spending compute on the first or partially converting a row.
        for index, rgb in enumerate(rgb_views):
            if (
                not isinstance(rgb, np.ndarray)
                or rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or min(rgb.shape[:2]) < 2
            ):
                raise ValueError(f"rgb_views[{index}] must be a uint8 HWC RGB array with 3 channels")
        result = []
        with _deterministic_float32(self._device.type):
            for index, rgb in enumerate(rgb_views):
                inputs = self._processor(
                    images=np.ascontiguousarray(rgb),
                    input_data_format="channels_last",
                    data_format="channels_first",
                    return_tensors="pt",
                    **self._processor_settings,
                )
                pixel_values = inputs["pixel_values"].to(device=self._device, dtype=torch.float32)
                if min(pixel_values.shape[-2:]) < 14:
                    raise ValueError(f"rgb_views[{index}] has an unsupported extreme aspect ratio")
                prediction = self._model(pixel_values=pixel_values).predicted_depth
                if prediction.ndim != 3 or prediction.shape[0] != 1:
                    raise ValueError(f"Unexpected Depth Anything output shape: {tuple(prediction.shape)}")
                depth = prediction[0].detach().to(device="cpu", dtype=torch.float32).numpy()
                if not np.isfinite(depth).all():
                    raise ValueError(f"Depth Anything prediction for view {index} contains non-finite pixels")
                result.append(normalize_relative_depth(
                    depth,
                    percentiles=self._config["percentiles"],
                    image_size=self._config["image_size"],
                    source_shape=rgb.shape[:2],
                ))
        return np.stack(result, axis=0)

    def provenance(self) -> dict[str, Any]:
        """Return stable artifact identity, processor choices and library versions.

        No hostnames, cache paths, timestamps or device IDs enter this record.
        Hashes are read from cached files only; inference has already loaded
        the pinned artifacts. Comparing records detects offline/online drift.
        """
        if self._provenance is None:
            from transformers.utils.hub import cached_file

            files = {}
            for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
                path = cached_file(
                    self._config["model_id"],
                    filename,
                    revision=self._config["revision"],
                    local_files_only=True,
                )
                if path is None:
                    raise FileNotFoundError(f"Loaded Depth Anything artifact is not cached: {filename}")
                files[filename] = _sha256(path)
            self._provenance = {
                "schema_version": 1,
                "config": self.config,
                "processor": {
                    "class": type(self._processor).__name__,
                    "use_fast": False,
                    "input_data_format": "channels_last",
                    "data_format": "channels_first",
                    **deepcopy(self._processor_settings),
                },
                "files_sha256": files,
                "libraries": {name: version(name) for name in (
                    "numpy", "torch", "transformers", "pillow", "huggingface-hub"
                )},
                "implementation_sha256": _sha256(__file__),
            }
        return deepcopy(self._provenance)
