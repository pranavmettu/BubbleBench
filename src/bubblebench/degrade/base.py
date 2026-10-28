"""Shared interface for the synthetic degradations.

A degradation is a *stage*: it takes a stream of frames and yields the degraded stream one
frame at a time, so a whole clip is never held in memory. A stage is a small frozen
dataclass holding a ``severity`` from 0 to 1 plus any fixed settings, which also makes it
easy to record in a results table. Frames are RGB uint8 arrays of shape (H, W, 3), and a
stage never modifies the frames it is given.
"""

import zlib
from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class Degradation(Protocol):
    """What a stage must provide to be used with :func:`degrade`.

    Contract: severity 0 yields frames equal to the input; the same ``seed`` always gives
    the same output; one output frame is produced per input frame, pulling input lazily.
    """

    @property
    def name(self) -> str:
        """Short unique label, such as ``"bubbles"``. Used to derive the stage's seed."""
        ...

    @property
    def severity(self) -> float:
        """How strong the degradation is: 0 leaves frames untouched, 1 is the worst."""
        ...

    def apply(self, frames: Iterable[Frame], fps: float, seed: int) -> Iterator[Frame]:
        """Yield the degraded version of each frame, in order."""
        ...


def validate_severity(severity: float) -> None:
    """Raise ValueError unless ``severity`` is in [0, 1]."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")


def stage_seed(seed: int, name: str) -> int:
    """Seed for one stage, derived from the master seed and the stage's name.

    It depends on the name, not the stage's position in the list, so adding, removing, or
    reordering stages never changes the random draws of the others.
    """
    sequence = np.random.SeedSequence([seed, zlib.crc32(name.encode())])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def degrade(
    frames: Iterable[Frame],
    fps: float,
    stages: Sequence[Degradation],
    seed: int,
) -> Iterator[Frame]:
    """Pass ``frames`` through ``stages`` in list order and yield the degraded frames.

    Args:
        frames: Clean RGB frames, each shape (H, W, 3), dtype uint8.
        fps: Frames per second of the clip (> 0). Stages use it to turn rates in Hz or
            pixels per second into per-frame amounts.
        stages: Degradations to apply, first to last.
        seed: Master seed (>= 0). The same seed on the same input gives identical output.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"stage names must be unique, got {names}")

    stream: Iterable[Frame] = frames
    for stage in stages:
        stream = stage.apply(stream, fps, stage_seed(seed, stage.name))
    return iter(stream)
