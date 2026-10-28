"""Turbidity: light attenuation and veiling backscatter in murky water.

Uses the standard underwater image formation model ``I = J * t + B * (1 - t)``,
where ``J`` is the clean frame, ``B`` the veiling (backscatter) color, and
``t = exp(-beta * depth)`` the transmission. Larger ``beta`` means murkier water.
"""

import numpy as np
from numpy.typing import NDArray

# Greenish-blue veiling light typical of coastal and river water, in [0, 1] RGB.
DEFAULT_VEIL_COLOR: tuple[float, float, float] = (0.25, 0.45, 0.40)


def apply_turbidity(
    frame: NDArray[np.uint8],
    beta: float,
    depth: NDArray[np.floating] | float = 1.0,
    veil_color: tuple[float, float, float] = DEFAULT_VEIL_COLOR,
) -> NDArray[np.uint8]:
    """Return ``frame`` degraded by turbid water.

    Args:
        frame: Clean RGB frame, shape (H, W, 3), dtype uint8.
        beta: Attenuation coefficient (>= 0). 0 leaves the frame unchanged.
        depth: Scene distance per pixel, shape (H, W), or a single scalar.
        veil_color: Backscatter color as RGB in [0, 1].
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 array of shape (H, W, 3)")
    if beta < 0:
        raise ValueError("beta must be >= 0")

    depth_arr = np.broadcast_to(np.asarray(depth, dtype=np.float64), frame.shape[:2])
    if (depth_arr < 0).any():
        raise ValueError("depth must be >= 0")

    transmission = np.exp(-beta * depth_arr)[..., None]
    veil = np.asarray(veil_color, dtype=np.float64) * 255.0
    degraded = frame * transmission + veil * (1.0 - transmission)
    return np.clip(np.rint(degraded), 0, 255).astype(np.uint8)
