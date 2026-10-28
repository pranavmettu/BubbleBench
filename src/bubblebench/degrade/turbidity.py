"""Turbidity: light attenuation, veiling backscatter, and forward-scatter blur.

Uses the standard underwater image formation model
``I = blur(J) * t + B * (1 - t)``, where ``J`` is the clean frame, ``B`` the veiling
(backscatter) color, and ``t = exp(-beta * depth)`` the transmission. Larger ``beta``
means murkier water. The blur models forward scatter: light from a distant object
spreads out on its way to the camera, so far objects look soft as well as faded.
It is off unless ``blur_strength`` is set. ``apply_turbidity`` works on one frame;
``Turbidity`` wraps it as a stage for :func:`bubblebench.degrade.base.degrade`.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from bubblebench.degrade.base import Frame, validate_severity

# Greenish-blue veiling light typical of coastal and river water, in [0, 1] RGB.
DEFAULT_VEIL_COLOR: tuple[float, float, float] = (0.25, 0.45, 0.40)

# Number of uniformly blurred copies blended when depth varies across the frame.
_BLUR_LEVELS = 6


def apply_turbidity(
    frame: NDArray[np.uint8],
    beta: float,
    depth: NDArray[np.floating] | float = 1.0,
    veil_color: tuple[float, float, float] = DEFAULT_VEIL_COLOR,
    blur_strength: float = 0.0,
) -> NDArray[np.uint8]:
    """Return ``frame`` degraded by turbid water.

    Args:
        frame: Clean RGB frame, shape (H, W, 3), dtype uint8.
        beta: Attenuation coefficient (>= 0). 0 leaves the frame unchanged.
        depth: Scene distance per pixel, shape (H, W), or a single scalar.
        veil_color: Backscatter color as RGB in [0, 1].
        blur_strength: Forward-scatter blur, in pixels of Gaussian sigma per unit of
            optical depth (``beta * depth``). 0 disables the blur. Sigma grows linearly
            with distance, which is a simplification of real small-angle scattering.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 array of shape (H, W, 3)")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if blur_strength < 0:
        raise ValueError("blur_strength must be >= 0")

    depth_arr = np.broadcast_to(np.asarray(depth, dtype=np.float64), frame.shape[:2])
    if (depth_arr < 0).any():
        raise ValueError("depth must be >= 0")

    signal = frame.astype(np.float64)
    if blur_strength > 0:
        signal = _depth_varying_blur(signal, blur_strength * beta * depth_arr)

    transmission = np.exp(-beta * depth_arr)[..., None]
    veil = np.asarray(veil_color, dtype=np.float64) * 255.0
    degraded = signal * transmission + veil * (1.0 - transmission)
    return np.clip(np.rint(degraded), 0, 255).astype(np.uint8)


# Settings at severity 1. Placeholders until the Gate C realism review sets them.
BETA_MAX = 2.5
BLUR_STRENGTH = 3.0


@dataclass(frozen=True)
class Turbidity:
    """Pipeline stage for turbid water (see :class:`bubblebench.degrade.base.Degradation`).

    Severity sets ``beta`` linearly from 0 to ``BETA_MAX``. ``blur_strength`` is a property
    of the water and stays fixed, so the blur widens with severity through ``beta``.
    The stage has no state over time, so ``fps`` and ``seed`` are unused.
    """

    severity: float
    blur_strength: float = BLUR_STRENGTH
    name: ClassVar[str] = "turbidity"

    def __post_init__(self) -> None:
        validate_severity(self.severity)
        if self.blur_strength < 0:
            raise ValueError("blur_strength must be >= 0")

    def apply(self, frames: Iterable[Frame], fps: float, seed: int) -> Iterator[Frame]:
        """Yield each frame degraded by turbid water."""
        beta = self.severity * BETA_MAX
        for frame in frames:
            yield apply_turbidity(frame, beta, blur_strength=self.blur_strength)


def _gaussian_kernel(sigma: float) -> NDArray[np.float64]:
    """1D Gaussian weights out to 3 sigma, normalized to sum to 1."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def _blur_first_axis(
    image: NDArray[np.float64], kernel: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convolve ``image`` with ``kernel`` along axis 0, repeating edge pixels at the border."""
    radius = len(kernel) // 2
    pad_width = [(radius, radius)] + [(0, 0)] * (image.ndim - 1)
    padded = np.pad(image, pad_width, mode="edge")
    out = np.zeros_like(image)
    for shift, weight in enumerate(kernel):
        out += weight * padded[shift : shift + image.shape[0]]
    return out


def _gaussian_blur(image: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Blur an (H, W, C) float image with a separable Gaussian of the given sigma."""
    if sigma <= 0:
        return image
    kernel = _gaussian_kernel(sigma)
    blurred = _blur_first_axis(image, kernel)  # vertical pass
    blurred = _blur_first_axis(blurred.swapaxes(0, 1), kernel)  # horizontal pass
    return blurred.swapaxes(0, 1)


def _depth_varying_blur(
    image: NDArray[np.float64], sigma_map: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Blur each pixel by its own sigma.

    Blurs the whole frame at a few evenly spaced sigmas, then for every pixel blends the
    two copies on either side of that pixel's sigma. This approximates a true per-pixel
    Gaussian, and costs one blur when sigma is the same everywhere.
    """
    sigma_max = float(sigma_map.max())
    if sigma_max == float(sigma_map.min()):
        return _gaussian_blur(image, sigma_max)

    step = sigma_max / (_BLUR_LEVELS - 1)
    position = sigma_map / step  # where each pixel falls between the blurred copies
    result = np.zeros_like(image)
    for level in range(_BLUR_LEVELS):
        weight = np.clip(1.0 - np.abs(position - level), 0.0, 1.0)[..., None]
        if weight.any():
            result += weight * _gaussian_blur(image, level * step)
    return result
