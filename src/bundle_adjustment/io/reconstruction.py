"""Reader for the pipeline's initialized-reconstruction interchange artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..models import BundleProblem, Camera, CameraIntrinsics, Observation, Pose

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "initialized_reconstruction"


def _read_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "reconstruction.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Could not read reconstruction metadata: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid reconstruction metadata: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("reconstruction metadata must be a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported reconstruction schema {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Artifact is not an initialized reconstruction")
    if metadata.get("coordinate_convention") != "world_to_camera":
        raise ValueError("Initialized reconstruction must use the world_to_camera convention")
    return metadata


def _required_array(
    arrays: np.lib.npyio.NpzFile, name: str, *, dtype: npt.DTypeLike
) -> npt.NDArray[np.generic]:
    if name not in arrays.files:
        raise ValueError(f"reconstruction.npz is missing required array {name!r}")
    return np.asarray(arrays[name], dtype=dtype)


def _string_identifiers(
    values: npt.NDArray[np.generic], name: str, expected: int
) -> tuple[str, ...]:
    if values.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {values.shape}")
    identifiers = tuple(str(value) for value in values)
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must contain unique, non-empty strings")
    return identifiers


def load_initialized_reconstruction(directory: str | Path) -> BundleProblem:
    """Load a schema-v1 initialized reconstruction as a validated bundle problem.

    This validates the contract at the process boundary. Array names and the
    pose convention are deliberately not guessed from older result artifacts.
    """
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Initialized reconstruction directory does not exist: {path}")
    metadata = _read_metadata(path)
    artifact_path = path / "reconstruction.npz"
    try:
        artifact = np.load(artifact_path, allow_pickle=False)
    except OSError as error:
        raise OSError(f"Could not read reconstruction artifact: {artifact_path}") from error
    with artifact:
        rotations = _required_array(artifact, "camera_rotations", dtype=np.float64)
        translations = _required_array(artifact, "camera_translations", dtype=np.float64)
        intrinsics = _required_array(artifact, "camera_intrinsics", dtype=np.float64)
        camera_identifiers = _required_array(artifact, "camera_identifiers", dtype=np.str_)
        points = _required_array(artifact, "points", dtype=np.float64)
        point_identifiers = _required_array(artifact, "point_identifiers", dtype=np.str_)
        camera_indices = _required_array(artifact, "observation_camera_indices", dtype=np.int64)
        point_indices = _required_array(artifact, "observation_point_indices", dtype=np.int64)
        observation_xy = _required_array(artifact, "observation_xy", dtype=np.float64)
        weights = _required_array(artifact, "observation_weights", dtype=np.float64)

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"camera_rotations must have shape (C, 3, 3), got {rotations.shape}")
    camera_count = rotations.shape[0]
    if camera_count == 0:
        raise ValueError("initialized reconstruction has no cameras")
    if translations.shape != (camera_count, 3):
        raise ValueError(
            f"camera_translations must have shape ({camera_count}, 3), got {translations.shape}"
        )
    if intrinsics.shape != (camera_count, 4):
        raise ValueError(
            f"camera_intrinsics must have shape ({camera_count}, 4), got {intrinsics.shape}"
        )
    identifiers = _string_identifiers(camera_identifiers, "camera_identifiers", camera_count)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
        raise ValueError(f"points must have shape (P, 3) with P > 0, got {points.shape}")
    point_count = points.shape[0]
    _string_identifiers(point_identifiers, "point_identifiers", point_count)
    observation_count = camera_indices.shape[0]
    if (
        point_indices.shape != (observation_count,)
        or observation_xy.shape != (observation_count, 2)
        or weights.shape != (observation_count,)
    ):
        raise ValueError("observation arrays must share a valid M-length leading dimension")
    if observation_count == 0:
        raise ValueError("initialized reconstruction has no observations")
    if np.any(camera_indices < 0) or np.any(camera_indices >= camera_count):
        raise ValueError("observation_camera_indices contains an out-of-range index")
    if np.any(point_indices < 0) or np.any(point_indices >= point_count):
        raise ValueError("observation_point_indices contains an out-of-range index")

    cameras = tuple(
        Camera(
            intrinsics=CameraIntrinsics(*[float(value) for value in intrinsics[index]]),
            pose=Pose(rotation=rotations[index], translation=translations[index]),
            identifier=identifiers[index],
        )
        for index in range(camera_count)
    )
    observations = tuple(
        Observation(
            camera_index=int(camera_indices[index]),
            point_index=int(point_indices[index]),
            xy=observation_xy[index],
            weight=float(weights[index]),
        )
        for index in range(observation_count)
    )
    problem_name = metadata.get("problem_name")
    return BundleProblem(
        cameras=cameras,
        points=points,
        observations=observations,
        name=problem_name if isinstance(problem_name, str) and problem_name else path.name,
    )
