"""Versioned, machine-readable optimization result artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from ..models import OptimizationResult

SCHEMA_VERSION = 1


def _package_version() -> str:
    try:
        return version("bundle-adjustment")
    except PackageNotFoundError:
        return "unknown"


def result_metadata(result: OptimizationResult) -> dict[str, Any]:
    """Return JSON-compatible, versioned metadata for an optimization result."""
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": _package_version(),
        "problem_name": result.problem.name,
        "counts": {
            "cameras": result.problem.num_cameras,
            "points": result.problem.num_points,
            "observations": len(result.problem.observations),
        },
        "termination_reason": result.termination_reason,
        "converged": result.converged,
        "diagnostics": asdict(result.diagnostics),
    }


def save_result(
    result: OptimizationResult, output_directory: str | Path, *, overwrite: bool = False
) -> Path:
    """Write result arrays and metadata, refusing accidental overwrites by default."""
    directory = Path(output_directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Output path is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ValueError(
            f"Output directory is not empty: {directory}; pass overwrite=True to replace files"
        )
    directory.mkdir(parents=True, exist_ok=True)
    problem = result.problem
    arrays_path = directory / "result.npz"
    metadata_path = directory / "result.json"
    with arrays_path.open("wb") as output:
        np.savez_compressed(
            output,
            camera_rotations=np.stack([camera.pose.rotation for camera in problem.cameras]),
            camera_translations=np.stack([camera.pose.translation for camera in problem.cameras]),
            camera_intrinsics=np.array(
                [
                    [
                        camera.intrinsics.fx,
                        camera.intrinsics.fy,
                        camera.intrinsics.cx,
                        camera.intrinsics.cy,
                    ]
                    for camera in problem.cameras
                ],
                dtype=np.float64,
            ),
            camera_identifiers=np.array(
                [camera.identifier or "" for camera in problem.cameras], dtype=np.str_
            ),
            points=problem.points,
            observation_camera_indices=np.array(
                [observation.camera_index for observation in problem.observations], dtype=np.int64
            ),
            observation_point_indices=np.array(
                [observation.point_index for observation in problem.observations], dtype=np.int64
            ),
            observation_xy=np.stack([observation.xy for observation in problem.observations]),
            observation_weights=np.array(
                [observation.weight for observation in problem.observations], dtype=np.float64
            ),
        )
    metadata_path.write_text(
        json.dumps(result_metadata(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def read_result_metadata(output_directory: str | Path) -> dict[str, Any]:
    """Read and validate the JSON metadata stored by :func:`save_result`."""
    path = Path(output_directory) / "result.json"
    try:
        metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Could not read result metadata: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON result metadata: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Result metadata must be a JSON object: {path}")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported result schema {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return metadata
