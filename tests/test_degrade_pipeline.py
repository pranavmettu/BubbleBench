"""Tests for chaining stages with degrade() and for how stage seeds are derived."""

from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pytest

from bubblebench.degrade import Frame, Turbidity, degrade
from bubblebench.degrade.base import stage_seed


class _Stage:
    """Minimal stage that applies ``op`` to each frame and records the seed and fps it got."""

    severity = 0.5

    def __init__(self, name: str, op: Callable[[Frame], Frame]) -> None:
        self.name = name
        self.op = op
        self.seed: int | None = None
        self.fps: float | None = None

    def apply(self, frames: Iterable[Frame], fps: float, seed: int) -> Iterator[Frame]:
        self.seed, self.fps = seed, fps
        return (self.op(frame) for frame in frames)


def _plus_ten(name: str = "plus_ten") -> _Stage:
    return _Stage(name, lambda frame: frame + 10)


def _times_two(name: str = "times_two") -> _Stage:
    return _Stage(name, lambda frame: frame * 2)


def _flat_frames() -> list[Frame]:
    return [np.full((4, 4, 3), 5, dtype=np.uint8) for _ in range(3)]


def test_stages_apply_in_list_order() -> None:
    add_then_scale = list(degrade(_flat_frames(), 30.0, [_plus_ten(), _times_two()], seed=0))
    scale_then_add = list(degrade(_flat_frames(), 30.0, [_times_two(), _plus_ten()], seed=0))
    assert (add_then_scale[0] == (5 + 10) * 2).all()
    assert (scale_then_add[0] == 5 * 2 + 10).all()


def test_no_stages_passes_frames_through() -> None:
    frames = _flat_frames()
    out = list(degrade(frames, 30.0, [], seed=0))
    assert len(out) == len(frames)
    for a, b in zip(out, frames, strict=True):
        np.testing.assert_array_equal(a, b)


def test_stage_seed_does_not_depend_on_position() -> None:
    first, second = _plus_ten("a"), _times_two("b")
    degrade(_flat_frames(), 30.0, [first, second], seed=7)
    swapped_first, swapped_second = _times_two("b"), _plus_ten("a")
    degrade(_flat_frames(), 30.0, [swapped_first, swapped_second], seed=7)
    assert first.seed == swapped_second.seed
    assert second.seed == swapped_first.seed


def test_stage_seed_depends_on_name_and_master_seed() -> None:
    assert stage_seed(0, "bubbles") != stage_seed(0, "flicker")
    assert stage_seed(0, "bubbles") != stage_seed(1, "bubbles")
    assert stage_seed(0, "bubbles") == stage_seed(0, "bubbles")


def test_stage_seeds_are_pinned() -> None:
    # Changing how seeds are derived would silently change every past result.
    assert stage_seed(0, "bubbles") == 7441772433222424487
    assert stage_seed(0, "turbidity") == 665705182734642035
    assert stage_seed(1, "bubbles") == 4265058675248666100


def test_stages_receive_the_clip_fps() -> None:
    stage = _plus_ten()
    degrade(_flat_frames(), 24.0, [stage], seed=0)
    assert stage.fps == 24.0


def test_rejects_duplicate_stage_names() -> None:
    with pytest.raises(ValueError):
        degrade(_flat_frames(), 30.0, [_plus_ten("same"), _times_two("same")], seed=0)


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_rejects_non_positive_fps(fps: float) -> None:
    with pytest.raises(ValueError):
        degrade(_flat_frames(), fps, [_plus_ten()], seed=0)


def test_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        degrade(_flat_frames(), 30.0, [_plus_ten()], seed=-1)


def test_chain_pulls_frames_lazily() -> None:
    pulled: list[int] = []

    def source() -> Iterator[Frame]:
        for frame in _flat_frames():
            pulled.append(1)
            yield frame

    stream = degrade(source(), 30.0, [_plus_ten(), _times_two()], seed=0)
    next(stream)
    assert len(pulled) == 1


def test_turbidity_through_degrade_is_reproducible_and_severity_matters() -> None:
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8) for _ in range(3)]
    stages = [Turbidity(severity=0.5)]
    first = list(degrade(frames, 30.0, stages, seed=3))
    second = list(degrade(frames, 30.0, stages, seed=3))
    clean = list(degrade(frames, 30.0, [Turbidity(severity=0.0)], seed=3))
    for a, b, c, original in zip(first, second, clean, frames, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(c, original)
        assert not np.array_equal(a, original)
