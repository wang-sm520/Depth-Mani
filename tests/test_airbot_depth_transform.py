from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airbot_depth.depth import (
    DEFAULT_DEPTH_CONFIG,
    DepthAnythingTransform,
    canonical_depth_config,
    normalize_relative_depth,
)


def test_far_zero_is_valid_and_letterbox_padding_is_invalid():
    depth = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32)
    result = normalize_relative_depth(depth, (0, 100), image_size=4)
    assert result.dtype == np.float32
    assert result.shape == (2, 4, 4)
    np.testing.assert_array_equal(result[:, [0, 3], :], 0)
    np.testing.assert_array_equal(result[1, 1:3, :], 1)
    np.testing.assert_array_equal(result[0, 1:3, 0], 0)
    np.testing.assert_array_equal(result[0, 1:3, -1], 1)


def test_positive_affine_change_does_not_change_normalized_depth():
    depth = np.arange(80, dtype=np.float64).reshape(8, 10)
    original = normalize_relative_depth(depth, (2, 98), image_size=12)
    affine = normalize_relative_depth(depth * 4.75 - 31, (2, 98), image_size=12)
    np.testing.assert_allclose(affine, original, atol=1e-6, rtol=0)
    assert original[0].min() >= 0
    assert original[0].max() <= 1


def test_output_uses_original_camera_aspect_after_patch_rounding():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = normalize_relative_depth(depth, (0, 100), image_size=8, source_shape=(3, 6))
    np.testing.assert_array_equal(result[1, 2:6], 1)
    np.testing.assert_array_equal(result[1, :2], 0)
    np.testing.assert_array_equal(result[1, 6:], 0)


def test_partial_invalid_pixels_are_masked_without_losing_far_zero():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    depth[1, 1] = np.nan
    depth[2, 2] = np.inf
    result = normalize_relative_depth(depth, (0, 100), image_size=4)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:, 1, 1], 0)
    np.testing.assert_array_equal(result[:, 2, 2], 0)
    assert result[0, 0, 0] == 0
    assert result[1, 0, 0] == 1
    np.testing.assert_array_equal(result[1].astype(bool), np.isfinite(depth))


@pytest.mark.parametrize("depth, message", [
    (np.full((4, 4), np.nan), "no finite valid pixels"),
    (np.full((4, 4), np.inf), "no finite valid pixels"),
    (np.zeros((4, 4)), "degenerate"),
    (np.full((4, 4), 3.25), "degenerate"),
    (np.zeros((4, 4, 1)), "2-D"),
    (np.empty((0, 4)), "2-D"),
])
def test_invalid_or_degenerate_predictions_raise(depth, message):
    with pytest.raises(ValueError, match=message):
        normalize_relative_depth(depth)


@pytest.mark.parametrize("override, message", [
    ({"revision": "main"}, "immutable"),
    ({"representation": "metric_depth"}, "representation"),
    ({"input_color": "BGR"}, "input_color"),
    ({"processor_use_fast": True}, "processor_use_fast"),
    ({"image_size": True}, "image_size"),
    ({"input_size": 512}, "multiple"),
    ({"percentiles": [98, 2]}, "percentiles"),
    ({"percentiles": [2, float("nan")]}, "percentiles"),
    ({"percentiles": [False, 98]}, "percentiles"),
    ({"unknown_option": True}, "Unknown"),
])
def test_bad_config_is_rejected_before_loading_model(override, message):
    with pytest.raises(ValueError, match=message):
        DepthAnythingTransform(override, local_files_only=True)


def test_config_is_complete_and_not_aliased():
    supplied = {"image_size": 64, "percentiles": [1, 99]}
    canonical = canonical_depth_config(supplied)
    supplied["percentiles"][0] = 50
    assert canonical["percentiles"] == [1.0, 99.0]
    assert canonical["image_size"] == 64
    assert canonical["revision"] == DEFAULT_DEPTH_CONFIG["revision"]
    assert canonical_depth_config(DEFAULT_DEPTH_CONFIG) == DEFAULT_DEPTH_CONFIG


def test_conflicting_cuda_workspace_is_rejected_before_model_loading(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        DepthAnythingTransform({}, device="cuda", local_files_only=True)


def _fake_transform(predictions=None):
    """Bypass weight loading, leaving input validation/postprocessing real."""
    transform = DepthAnythingTransform.__new__(DepthAnythingTransform)
    transform._config = canonical_depth_config({"image_size": 4, "percentiles": [0, 100]})
    transform._device = torch.device("cpu")
    transform._processor_settings = {}
    calls = []

    def processor(*, images, **kwargs):
        calls.append(images.copy())
        # Real processing always returns a multiple-of-14 model input.
        return {"pixel_values": torch.zeros((1, 3, 14, 28))}

    def model(**kwargs):
        if predictions is None:
            output = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        else:
            output = torch.from_numpy(predictions)
        return SimpleNamespace(predicted_depth=output)

    transform._processor = processor
    transform._model = model
    return transform, calls


@pytest.mark.parametrize("views", [
    [],
    np.zeros((2, 4, 4, 3), dtype=np.uint8),
    [np.zeros((4, 4, 3), dtype=np.float32)],
    [np.zeros((3, 4, 4), dtype=np.uint8)],
    [np.zeros((4, 4, 4), dtype=np.uint8)],
    [np.zeros((4, 4), dtype=np.uint8)],
    [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.float32)],
])
def test_bad_rgb_rejected_before_any_camera_inference(views):
    transform, calls = _fake_transform()
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        transform(views)
    assert not calls


def test_mixed_camera_sizes_preserve_order_and_return_two_channels():
    transform, calls = _fake_transform()
    first = np.full((12, 20, 3), 20, dtype=np.uint8)
    second = np.full((24, 16, 3), 70, dtype=np.uint8)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    result = transform([first, second])
    assert result.shape == (2, 2, 4, 4)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(calls[0], first)
    np.testing.assert_array_equal(calls[1], second)
    assert torch.are_deterministic_algorithms_enabled() == deterministic_before
    config = transform.config
    config["percentiles"][0] = 10
    assert transform.config["percentiles"] == [0, 100]


def test_model_nonfinite_prediction_is_rejected():
    depth = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    depth[0, 0, 0] = np.nan
    transform, _ = _fake_transform(depth)
    with pytest.raises(ValueError, match="non-finite"):
        transform([np.zeros((8, 8, 3), dtype=np.uint8)])
