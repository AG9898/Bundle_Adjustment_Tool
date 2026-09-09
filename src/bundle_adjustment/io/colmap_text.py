"""Strict importer for COLMAP's text sparse-model format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import BundleProblem, Camera, CameraIntrinsics, Observation, Pose


@dataclass(frozen=True, slots=True)
class _RawImage:
    identifier: int
    camera_identifier: int
    name: str
    pose: Pose
    observations: tuple[tuple[float, float, int], ...]


def _data_lines(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from a one-record-per-line file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError(f"Could not read COLMAP file: {path}") from error
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _parse_intrinsics(path: Path) -> dict[int, CameraIntrinsics]:
    cameras: dict[int, CameraIntrinsics] = {}
    for line_number, line in enumerate(_data_lines(path), start=1):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(
                f"{path.name}:{line_number}: expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"
            )
        camera_identifier = int(fields[0])
        model = fields[1]
        parameters = [float(value) for value in fields[4:]]
        if model == "PINHOLE" and len(parameters) == 4:
            intrinsics = CameraIntrinsics(*parameters)
        elif model == "SIMPLE_PINHOLE" and len(parameters) == 3:
            focal_length, cx, cy = parameters
            intrinsics = CameraIntrinsics(focal_length, focal_length, cx, cy)
        else:
            raise ValueError(
                f"{path.name}:{line_number}: unsupported COLMAP camera model {model!r}; "
                "only PINHOLE and SIMPLE_PINHOLE are currently supported"
            )
        if camera_identifier in cameras:
            raise ValueError(f"{path.name}:{line_number}: duplicate CAMERA_ID {camera_identifier}")
        cameras[camera_identifier] = intrinsics
    if not cameras:
        raise ValueError(f"{path.name}: no cameras found")
    return cameras


def _parse_images(path: Path) -> list[_RawImage]:
    """Parse paired image/point-observation lines without dropping empty tracks."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError(f"Could not read COLMAP file: {path}") from error
    images: list[_RawImage] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(
                f"{path.name}:{index + 1}: expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"
            )
        if index + 1 >= len(lines):
            raise ValueError(f"{path.name}:{index + 1}: missing POINTS2D line")
        try:
            image_identifier = int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = np.array([float(value) for value in fields[5:8]], dtype=np.float64)
            camera_identifier = int(fields[8])
            rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            pose = Pose(rotation=rotation, translation=translation)
        except ValueError as error:
            raise ValueError(f"{path.name}:{index + 1}: invalid image pose") from error
        observation_fields = lines[index + 1].split()
        if len(observation_fields) % 3 != 0:
            raise ValueError(
                f"{path.name}:{index + 2}: POINTS2D fields must occur in X Y POINT3D_ID triples"
            )
        observations = tuple(
            (
                float(observation_fields[offset]),
                float(observation_fields[offset + 1]),
                int(observation_fields[offset + 2]),
            )
            for offset in range(0, len(observation_fields), 3)
        )
        images.append(
            _RawImage(
                identifier=image_identifier,
                camera_identifier=camera_identifier,
                name=fields[9],
                pose=pose,
                observations=observations,
            )
        )
        index += 2
    if not images:
        raise ValueError(f"{path.name}: no images found")
    if len({image.identifier for image in images}) != len(images):
        raise ValueError(f"{path.name}: duplicate IMAGE_ID values")
    return images


def _parse_points(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(_data_lines(path), start=1):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path.name}:{line_number}: expected POINT3D_ID X Y Z ...")
        point_identifier = int(fields[0])
        if point_identifier in points:
            raise ValueError(f"{path.name}:{line_number}: duplicate POINT3D_ID {point_identifier}")
        coordinates = np.array([float(value) for value in fields[1:4]], dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"{path.name}:{line_number}: point coordinates must be finite")
        points[point_identifier] = coordinates
    if not points:
        raise ValueError(f"{path.name}: no 3D points found")
    return points


def load_colmap_text(model_directory: str | Path) -> BundleProblem:
    """Load a valid COLMAP text model into the library's world-to-camera contract.

    Untriangulated observations (``POINT3D_ID == -1``), missing point IDs, and
    images left with no valid observations are excluded. The remaining image and
    point IDs are remapped densely and deterministically in input order.
    """
    directory = Path(model_directory)
    if not directory.is_dir():
        raise ValueError(f"COLMAP model directory does not exist: {directory}")
    intrinsics_by_identifier = _parse_intrinsics(directory / "cameras.txt")
    raw_images = _parse_images(directory / "images.txt")
    points_by_identifier = _parse_points(directory / "points3D.txt")

    cameras: list[Camera] = []
    observations: list[Observation] = []
    point_identifier_to_index: dict[int, int] = {}
    ordered_points: list[np.ndarray] = []
    for raw_image in raw_images:
        intrinsics = intrinsics_by_identifier.get(raw_image.camera_identifier)
        if intrinsics is None:
            raise ValueError(
                f"images.txt: IMAGE_ID {raw_image.identifier} references unknown "
                f"CAMERA_ID {raw_image.camera_identifier}"
            )
        valid_measurements = [
            measurement
            for measurement in raw_image.observations
            if measurement[2] != -1 and measurement[2] in points_by_identifier
        ]
        if not valid_measurements:
            continue
        camera_index = len(cameras)
        cameras.append(
            Camera(
                intrinsics=intrinsics,
                pose=raw_image.pose,
                identifier=f"{raw_image.identifier}:{raw_image.name}",
            )
        )
        for x, y, point_identifier in valid_measurements:
            point_index = point_identifier_to_index.get(point_identifier)
            if point_index is None:
                point_index = len(ordered_points)
                point_identifier_to_index[point_identifier] = point_index
                ordered_points.append(points_by_identifier[point_identifier])
            observations.append(
                Observation(
                    camera_index=camera_index,
                    point_index=point_index,
                    xy=np.array([x, y], dtype=np.float64),
                )
            )
    if not cameras or not ordered_points or not observations:
        raise ValueError("COLMAP model contains no triangulated observations to optimize")
    return BundleProblem(
        cameras=tuple(cameras),
        points=np.vstack(ordered_points),
        observations=tuple(observations),
        name=directory.name,
    )
