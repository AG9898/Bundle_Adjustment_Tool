from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bundle_adjustment.cli import main
from bundle_adjustment.io import (
    load_colmap_text,
    load_initialized_reconstruction,
    read_result_metadata,
)


def _pixels(point: np.ndarray, translation_x: float) -> tuple[float, float]:
    point_camera = point + np.array([translation_x, 0.0, 0.0])
    return 800.0 * point_camera[0] / point_camera[2] + 320.0, 810.0 * point_camera[
        1
    ] / point_camera[2] + 240.0


def _write_colmap_fixture(directory: Path) -> None:
    points = {
        42: np.array([0.0, 0.0, 10.0]),
        77: np.array([1.0, 0.5, 12.0]),
        99: np.array([-0.8, 0.4, 9.0]),
        123: np.array([0.4, -0.7, 11.0]),
        300: np.array([-1.2, -0.4, 8.0]),
    }
    directory.mkdir()
    (directory / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 800 810 320 240\n", encoding="utf-8"
    )
    image_lines = ["# Image list"]
    for image_id, translation_x, camera_translation_x, name in (
        (1, 0.0, 0.0, "first.jpg"),
        (2, -1.0, -0.9, "second.jpg"),
    ):
        image_lines.append(f"{image_id} 1 0 0 0 {camera_translation_x} 0 0 1 {name}")
        observations = []
        for point_id, point in points.items():
            x, y = _pixels(point, translation_x)
            observations.extend((f"{x:.12f}", f"{y:.12f}", str(point_id)))
        image_lines.append(" ".join(observations))
    (directory / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = ["# 3D point list"]
    for point_id, point in points.items():
        point_lines.append(f"{point_id} {point[0]} {point[1]} {point[2]} 255 255 255 0")
    (directory / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def _write_initialized_reconstruction(directory: Path) -> None:
    points = np.array([[0.0, 0.0, 10.0], [1.0, 0.5, 12.0]], dtype=np.float64)
    directory.mkdir()
    np.savez_compressed(
        directory / "reconstruction.npz",
        camera_rotations=np.stack((np.eye(3), np.eye(3))),
        camera_translations=np.array([[0.0, 0.0, 0.0], [-0.9, 0.0, 0.0]]),
        camera_intrinsics=np.array([[800.0, 810.0, 320.0, 240.0]] * 2),
        camera_identifiers=np.array(["1:first.jpg", "2:second.jpg"]),
        points=points,
        point_identifiers=np.array(["42", "77"]),
        observation_camera_indices=np.array([0, 0, 1, 1]),
        observation_point_indices=np.array([0, 1, 0, 1]),
        observation_xy=np.array(
            [
                _pixels(points[0], 0.0),
                _pixels(points[1], 0.0),
                _pixels(points[0], -0.9),
                _pixels(points[1], -0.9),
            ]
        ),
        observation_weights=np.ones(4),
    )
    (directory / "reconstruction.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "initialized_reconstruction",
                "coordinate_convention": "world_to_camera",
                "problem_name": "triangulated-fixture",
            }
        ),
        encoding="utf-8",
    )


def test_colmap_text_loader_preserves_calibration_pose_and_dense_ids(tmp_path: Path) -> None:
    model_directory = tmp_path / "model"
    _write_colmap_fixture(model_directory)

    problem = load_colmap_text(model_directory)

    assert problem.num_cameras == 2
    assert problem.num_points == 5
    assert len(problem.observations) == 10
    assert problem.cameras[0].identifier == "1:first.jpg"
    np.testing.assert_allclose(problem.cameras[0].pose.translation, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(problem.cameras[1].pose.translation, [-0.9, 0.0, 0.0])
    assert problem.cameras[0].intrinsics.fx == 800.0
    assert {observation.point_index for observation in problem.observations} == set(range(5))


def test_cli_validates_optimizes_and_reports_versioned_artifacts(
    tmp_path: Path, capsys: object
) -> None:
    model_directory = tmp_path / "model"
    output_directory = tmp_path / "result"
    _write_colmap_fixture(model_directory)

    assert main(("validate", str(model_directory))) == 0
    validated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validated["observations"] == 10

    assert main(("optimize", str(model_directory), str(output_directory), "--loss", "squared")) == 0
    optimization = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert optimization["converged"]
    assert (output_directory / "result.npz").is_file()
    metadata = read_result_metadata(output_directory)
    assert metadata["schema_version"] == 1
    assert metadata["diagnostics"]["final_cost"] < metadata["diagnostics"]["initial_cost"]

    assert main(("report", str(output_directory), "--json")) == 0
    report = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert report["termination_reason"] == optimization["termination_reason"]


def test_initialized_reconstruction_is_validated_and_accepted_by_cli(
    tmp_path: Path, capsys: object
) -> None:
    directory = tmp_path / "initialized"
    _write_initialized_reconstruction(directory)

    problem = load_initialized_reconstruction(directory)

    assert problem.name == "triangulated-fixture"
    assert problem.num_cameras == 2
    assert problem.num_points == 2
    assert main(("validate", str(directory))) == 0
    validated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validated["problem_name"] == "triangulated-fixture"
