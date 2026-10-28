"""Tests for the turbidity degradation."""

import numpy as np
import pytest

from bubblebench.degrade import apply_turbidity


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)


def test_zero_beta_is_identity() -> None:
    frame = _frame()
    np.testing.assert_array_equal(apply_turbidity(frame, beta=0.0), frame)


def test_high_beta_converges_to_veil_color() -> None:
    out = apply_turbidity(_frame(), beta=50.0, veil_color=(1.0, 0.0, 0.5))
    assert (out == np.array([255, 0, 128], dtype=np.uint8)).all()


def test_contrast_decreases_monotonically_with_beta() -> None:
    frame = _frame()
    stds = [apply_turbidity(frame, beta=b).std() for b in (0.0, 0.5, 1.0, 2.0)]
    assert stds == sorted(stds, reverse=True)


def test_per_pixel_depth_changes_only_far_pixels() -> None:
    frame = _frame()
    depth = np.zeros((8, 8))
    depth[:, 4:] = 3.0
    out = apply_turbidity(frame, beta=1.0, depth=depth)
    np.testing.assert_array_equal(out[:, :4], frame[:, :4])
    assert not np.array_equal(out[:, 4:], frame[:, 4:])


def test_output_is_uint8_and_same_shape() -> None:
    out = apply_turbidity(_frame(), beta=0.7)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)


@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32)])
def test_rejects_bad_frames(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_turbidity(bad, beta=1.0)


def test_rejects_negative_beta() -> None:
    with pytest.raises(ValueError):
        apply_turbidity(_frame(), beta=-0.1)


def _noisy_frame() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


def _roughness(frame: np.ndarray) -> float:
    """Mean absolute difference between horizontally neighboring pixels; lower is softer."""
    return float(np.abs(np.diff(frame.astype(np.float64), axis=1)).mean())


def test_blur_without_attenuation_is_identity() -> None:
    frame = _noisy_frame()
    np.testing.assert_array_equal(apply_turbidity(frame, beta=0.0, blur_strength=5.0), frame)


def test_blur_softens_more_as_strength_grows() -> None:
    frame = _noisy_frame()
    rough = [
        _roughness(apply_turbidity(frame, beta=1.0, blur_strength=s)) for s in (0.0, 1.0, 2.0, 4.0)
    ]
    assert rough == sorted(rough, reverse=True)
    assert rough[-1] < rough[0]


def test_blur_leaves_flat_frame_unchanged() -> None:
    flat = np.full((16, 16, 3), 100, dtype=np.uint8)
    blurred = apply_turbidity(flat, beta=1.0, blur_strength=3.0)
    np.testing.assert_array_equal(blurred, apply_turbidity(flat, beta=1.0))


def test_blur_follows_per_pixel_depth() -> None:
    frame = _noisy_frame()
    depth = np.zeros((32, 32))
    depth[:, 16:] = 2.0
    blurred = apply_turbidity(frame, beta=1.0, depth=depth, blur_strength=2.0)
    unblurred = apply_turbidity(frame, beta=1.0, depth=depth)
    np.testing.assert_array_equal(blurred[:, :16], frame[:, :16])  # depth 0: sharp, untouched
    assert _roughness(blurred[:, 20:]) < _roughness(unblurred[:, 20:])


def test_blur_keeps_dtype_and_shape() -> None:
    out = apply_turbidity(_noisy_frame(), beta=0.7, blur_strength=2.0)
    assert out.dtype == np.uint8
    assert out.shape == (32, 32, 3)


def test_rejects_negative_blur_strength() -> None:
    with pytest.raises(ValueError):
        apply_turbidity(_frame(), beta=1.0, blur_strength=-1.0)
