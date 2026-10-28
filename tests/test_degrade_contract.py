"""Contract every degradation stage must meet, run against each registered stage."""

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from bubblebench.degrade import Degradation, Frame, Turbidity

# Register each new stage here; every test below then runs against it.
STAGES: list[Callable[[float], Degradation]] = [Turbidity]


def _frames(count: int = 4) -> list[Frame]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8) for _ in range(count)]


def _run(stage: Degradation, frames: list[Frame], seed: int = 0) -> list[Frame]:
    return list(stage.apply(iter(frames), 30.0, seed))


@pytest.mark.parametrize("make_stage", STAGES)
def test_severity_zero_returns_equal_frames(make_stage: Callable[[float], Degradation]) -> None:
    frames = _frames()
    for out, original in zip(_run(make_stage(0.0), frames), frames, strict=True):
        np.testing.assert_array_equal(out, original)


@pytest.mark.parametrize("make_stage", STAGES)
def test_same_seed_gives_identical_output(make_stage: Callable[[float], Degradation]) -> None:
    frames = _frames()
    first = _run(make_stage(0.7), frames, seed=5)
    second = _run(make_stage(0.7), frames, seed=5)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("severity", [0.0, 1.0])
@pytest.mark.parametrize("make_stage", STAGES)
def test_input_frames_are_never_modified_or_shared(
    make_stage: Callable[[float], Degradation], severity: float
) -> None:
    frames = _frames()
    snapshot = [frame.copy() for frame in frames]
    outputs = _run(make_stage(severity), frames)
    for frame, saved in zip(frames, snapshot, strict=True):
        np.testing.assert_array_equal(frame, saved)
    assert not any(np.shares_memory(out, frame) for out, frame in zip(outputs, frames, strict=True))


@pytest.mark.parametrize("make_stage", STAGES)
def test_one_output_frame_per_input_with_same_shape_and_dtype(
    make_stage: Callable[[float], Degradation],
) -> None:
    frames = _frames(count=3)
    outputs = _run(make_stage(0.5), frames)
    assert len(outputs) == 3
    assert all(out.shape == (12, 16, 3) and out.dtype == np.uint8 for out in outputs)


@pytest.mark.parametrize("make_stage", STAGES)
def test_frames_are_pulled_lazily(make_stage: Callable[[float], Degradation]) -> None:
    pulled: list[int] = []

    def source() -> Iterator[Frame]:
        for frame in _frames():
            pulled.append(1)
            yield frame

    stream = make_stage(0.5).apply(source(), 30.0, 0)
    next(stream)
    assert len(pulled) == 1


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
@pytest.mark.parametrize("make_stage", STAGES)
def test_rejects_severity_outside_unit_interval(
    make_stage: Callable[[float], Degradation], bad: float
) -> None:
    with pytest.raises(ValueError):
        make_stage(bad)
