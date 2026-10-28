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
