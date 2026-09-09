"""Public API for dependable sparse bundle adjustment."""

from .models import (
    BundleProblem,
    Camera,
    CameraIntrinsics,
    Observation,
    OptimizationOptions,
    OptimizationResult,
    Pose,
)
from .optimizer import optimize

__all__ = [
    "BundleProblem",
    "Camera",
    "CameraIntrinsics",
    "Observation",
    "OptimizationOptions",
    "OptimizationResult",
    "Pose",
    "optimize",
]

__version__ = "0.1.0a0"
