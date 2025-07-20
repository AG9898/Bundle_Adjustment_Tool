import numpy as np
import numpy.typing as npt
from typing import List, Tuple
import matplotlib.pyplot as plt

from ..data.camera_models import CameraModel
from ..data.observations import BundleAdjustmentData, CameraPose, Observation
from ..solvers.sparse_lm_solver import SparseLMSolver
from ..visualizations.plot_reprojection_error import plot_reprojection_errors
from ..visualizations.plot_cameras import plot_cameras_and_points
from ..core.residuals import compute_residuals, compute_reprojection_error


def create_synthetic_scene(
    num_cameras: int = 5,
    num_points: int = 50,
    noise_std: float = 1.0,
    random_seed: int = 42
) -> Tuple[BundleAdjustmentData, List[CameraPose], npt.NDArray[np.float64]]:
    """
    Create a synthetic bundle adjustment scene with known ground truth.
    
    Args:
        num_cameras: Number of cameras to create
        num_points: Number of 3D points to create
        noise_std: Standard deviation of Gaussian noise for observations
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (bundle_data, true_camera_poses, true_points_3d)
    """
    np.random.seed(random_seed)
    
    # Create camera model
    focal_length = 1000.0
    principal_point = (640.0, 480.0)
    camera_model = CameraModel(focal_length, principal_point)
    
    # Create true camera poses in a grid pattern
    true_camera_poses = []
    grid_size = int(np.ceil(np.sqrt(num_cameras)))
    
    for i in range(num_cameras):
        # Grid position
        row = i // grid_size
        col = i % grid_size
        
        # Translation: grid pattern with some height variation
        translation = np.array([
            col * 10.0,  # X: grid column
            row * 10.0,  # Y: grid row
            5.0 + np.sin(i) * 2.0  # Z: varying height
        ])
        
        # Rotation: look towards center with some variation
        center_direction = np.array([0, 0, 0]) - translation
        center_direction = center_direction / np.linalg.norm(center_direction)
        
        # Create rotation matrix (simplified)
        up = np.array([0, 0, 1])
        right = np.cross(center_direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, center_direction)
        
        rotation = np.column_stack([right, up, -center_direction])
        
        true_camera_poses.append(CameraPose(rotation, translation))
    
    # Create true 3D points in a box
    true_points_3d = np.random.uniform(
        low=[-20, -20, -10],
        high=[20, 20, 10],
        size=(num_points, 3)
    )
    
    # Generate observations by projecting points
    observations = []
    for i, camera_pose in enumerate(true_camera_poses):
        # Project all points to this camera
        projected_points = camera_model.project(true_points_3d, (camera_pose.rotation, camera_pose.translation))
        
        # Add observations for points that are visible (in front of camera)
        for j, point_2d in enumerate(projected_points):
            # Check if point is in front of camera
            point_cam = camera_pose.rotation @ (true_points_3d[j] - camera_pose.translation)
            if point_cam[2] > 0:  # Point is in front of camera
                # Add Gaussian noise to observation
                noisy_point = point_2d + np.random.normal(0, noise_std, 2)
                observations.append(Observation(i, j, noisy_point))
    
    # Create bundle adjustment data
    bundle_data = BundleAdjustmentData(
        camera_poses=true_camera_poses,
        points_3d=true_points_3d,
        observations=observations,
        camera_model=camera_model
    )
    
    return bundle_data, true_camera_poses, true_points_3d


def perturb_parameters(
    camera_poses: List[CameraPose],
    points_3d: npt.NDArray[np.float64],
    rotation_perturbation: float = 0.1,
    translation_perturbation: float = 1.0,
    point_perturbation: float = 2.0
) -> Tuple[List[CameraPose], npt.NDArray[np.float64]]:
    """
    Add perturbations to camera poses and 3D points for testing optimization.
    
    Args:
        camera_poses: Original camera poses
        points_3d: Original 3D points
        rotation_perturbation: Standard deviation of rotation perturbation (radians)
        translation_perturbation: Standard deviation of translation perturbation
        point_perturbation: Standard deviation of point perturbation
        
    Returns:
        Tuple of (perturbed_camera_poses, perturbed_points_3d)
    """
    # Perturb camera poses
    perturbed_camera_poses = []
    for pose in camera_poses:
        # Perturb rotation using small angle approximation
        rotation_perturbation_vector = np.random.normal(0, rotation_perturbation, 3)
        rotation_perturbation_matrix = np.eye(3) + np.array([
            [0, -rotation_perturbation_vector[2], rotation_perturbation_vector[1]],
            [rotation_perturbation_vector[2], 0, -rotation_perturbation_vector[0]],
            [-rotation_perturbation_vector[1], rotation_perturbation_vector[0], 0]
        ])
        
        new_rotation = rotation_perturbation_matrix @ pose.rotation
        
        # Perturb translation
        translation_perturbation_vector = np.random.normal(0, translation_perturbation, 3)
        new_translation = pose.translation + translation_perturbation_vector
        
        perturbed_camera_poses.append(CameraPose(new_rotation, new_translation))
    
    # Perturb 3D points
    point_perturbation_vector = np.random.normal(0, point_perturbation, points_3d.shape)
    perturbed_points_3d = points_3d + point_perturbation_vector
    
    return perturbed_camera_poses, perturbed_points_3d


def test_synthetic_bundle_adjustment() -> None:
    """
    Test bundle adjustment on synthetic data with comprehensive validation.
    """
    print("=== Synthetic Bundle Adjustment Test ===")
    
    # Create synthetic scene
    print("Creating synthetic scene...")
    bundle_data, true_camera_poses, true_points_3d = create_synthetic_scene(
        num_cameras=5,
        num_points=50,
        noise_std=1.0,
        random_seed=42
    )
    
    print(f"Created scene with {len(bundle_data.camera_poses)} cameras, "
          f"{bundle_data.points_3d.shape[0]} points, {len(bundle_data.observations)} observations")
    
    # Perturb parameters for testing
    print("Adding perturbations to parameters...")
    perturbed_camera_poses, perturbed_points_3d = perturb_parameters(
        bundle_data.camera_poses,
        bundle_data.points_3d,
        rotation_perturbation=0.1,
        translation_perturbation=1.0,
        point_perturbation=2.0
    )
    
    # Update bundle data with perturbed parameters
    bundle_data.camera_poses = perturbed_camera_poses
    bundle_data.points_3d = perturbed_points_3d
    
    # Compute initial residuals
    initial_residuals = compute_residuals(bundle_data, perturbed_camera_poses, perturbed_points_3d)
    initial_residual_norm = compute_reprojection_error(bundle_data, perturbed_camera_poses, perturbed_points_3d)
    
    print(f"Initial residual norm: {initial_residual_norm:.6f}")
    
    # Plot initial state
    print("Plotting initial state...")
    plot_reprojection_errors(initial_residuals, "Initial Reprojection Errors")
    plot_cameras_and_points(perturbed_camera_poses, perturbed_points_3d, "Initial State")
    
    # Run bundle adjustment
    print("Running bundle adjustment...")
    solver = SparseLMSolver(
        data=bundle_data,
        max_iterations=20,
        initial_damping=1.0,
        damping_factor=10.0,
        convergence_threshold=1e-6
    )
    
    optimized_camera_poses, optimized_points_3d, final_residual_norm = solver.run()
    
    # Compute final residuals
    final_residuals = compute_residuals(bundle_data, optimized_camera_poses, optimized_points_3d)
    
    print(f"Final residual norm: {final_residual_norm:.6f}")
    print(f"Improvement: {initial_residual_norm - final_residual_norm:.6f}")
    
    # Plot final state
    print("Plotting final state...")
    plot_reprojection_errors(final_residuals, "Final Reprojection Errors")
    plot_cameras_and_points(optimized_camera_poses, optimized_points_3d, "Optimized State")
    
    # Validation assertions
    print("\n=== Validation ===")
    
    # 1. Residual norm should decrease
    assert final_residual_norm < initial_residual_norm, \
        f"Residual norm should decrease: {final_residual_norm} >= {initial_residual_norm}"
    print("✓ Residual norm decreased")
    
    # 2. 3D points should converge to reasonable values
    point_convergence_norm = np.linalg.norm(optimized_points_3d - true_points_3d)
    assert point_convergence_norm < 5.0, \
        f"3D points should converge: norm = {point_convergence_norm}"
    print(f"✓ 3D points converged (norm: {point_convergence_norm:.3f})")
    
    # 3. Camera centers should shift minimally
    camera_center_shifts = []
    for i, (true_pose, opt_pose) in enumerate(zip(true_camera_poses, optimized_camera_poses)):
        center_shift = np.linalg.norm(true_pose.translation - opt_pose.translation)
        camera_center_shifts.append(center_shift)
        assert center_shift < 3.0, \
            f"Camera {i} center shifted too much: {center_shift}"
    
    max_camera_shift = max(camera_center_shifts)
    print(f"✓ Camera centers stable (max shift: {max_camera_shift:.3f})")
    
    # 4. Final residual should be small
    assert final_residual_norm < 2.0, \
        f"Final residual should be small: {final_residual_norm}"
    print(f"✓ Final residual acceptable: {final_residual_norm:.3f}")
    
    print("\n=== Test Summary ===")
    print(f"Initial residual norm: {initial_residual_norm:.6f}")
    print(f"Final residual norm: {final_residual_norm:.6f}")
    print(f"Improvement: {initial_residual_norm - final_residual_norm:.6f}")
    print(f"3D point convergence norm: {point_convergence_norm:.6f}")
    print(f"Max camera center shift: {max_camera_shift:.6f}")
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_synthetic_bundle_adjustment() 