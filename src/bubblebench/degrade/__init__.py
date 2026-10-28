"""Synthetic underwater degradations applied to clean video frames."""

from bubblebench.degrade.base import Degradation, Frame, degrade
from bubblebench.degrade.turbidity import Turbidity, apply_turbidity

__all__ = ["Degradation", "Frame", "Turbidity", "apply_turbidity", "degrade"]
