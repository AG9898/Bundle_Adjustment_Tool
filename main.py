#!/usr/bin/env python3
"""
Bundle Adjustment Tool - Main Execution Script

This script demonstrates the end-to-end capability of the bundle adjustment library
using either synthetic data generation or real photogrammetric data from COLMAP.

Usage:
    # Synthetic data (default)
    python main.py --dataset synthetic
    
    # COLMAP data
    python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, List
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# Import bundle adjustment components
from src.data.camera_models import CameraModel
from src.data.observations import BundleAdjustmentData, CameraPose, Observation
from src.data.io_utils import load_colmap_bundle, print_colmap_summary, validate_colmap_data
from src.solvers.sparse_lm_solver import SparseLMSolver
from src.core.residuals import compute_residuals, compute_reprojection_error
from src.visualizations.plot_reprojection_error import plot_reprojection_errors
from src.visualizations.plot_cameras import plot_cameras_and_points


def create_synthetic_dataset(
    num_cameras: int = 6,
    num_points: int = 100,
    noise_std: float = 1.0,
    random_seed: int = 42
) -> BundleAdjustmentData:
    """
    Create a synthetic bundle adjustment dataset with known ground truth.
    
    Args:
        num_cameras: Number of cameras to create
        num_points: Number of 3D points to create
        noise_std: Standard deviation of Gaussian noise for observations
        random_seed: Random seed for reproducibility
        
    Returns:
        BundleAdjustmentData instance with perturbed initial poses and points
    """
    np.random.seed(random_seed)
    
    print(f"Creating synthetic dataset: {num_cameras} cameras, {num_points} points")
    
    # Create camera model
    focal_length = 1000.0
    principal_point = (640.0, 480.0)
    camera_model = CameraModel(focal_length, principal_point)
    
    # Create true camera poses in a circular pattern
    true_camera_poses = []
    for i in range(num_cameras):
        # Circular arrangement
        angle = i * 2 * np.pi / num_cameras
        radius = 15.0
        
        # Translation: circular pattern with height variation
        translation = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            8.0 + np.sin(i * 0.5) * 3.0  # Varying height
        ])
        
        # Rotation: look towards center with some variation
        center_direction = np.array([0, 0, 0]) - translation
        center_direction = center_direction / np.linalg.norm(center_direction)
        
        # Create rotation matrix
        up = np.array([0, 0, 1])
        right = np.cross(center_direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, center_direction)
        
        rotation = np.column_stack([right, up, -center_direction])
        
        true_camera_poses.append(CameraPose(rotation, translation))
    
    # Create true 3D points in a structured pattern
    true_points_3d = np.zeros((num_points, 3))
    
    # Create points in a structured pattern (cube + some random points)
    n_structured = min(num_points, 80)
    n_random = num_points - n_structured
    
    # Structured points in a cube
    if n_structured > 0:
        cube_points = np.random.uniform(
            low=[-12, -12, -8],
            high=[12, 12, 8],
            size=(n_structured, 3)
        )
        true_points_3d[:n_structured] = cube_points
    
    # Random points for additional coverage
    if n_random > 0:
        random_points = np.random.uniform(
            low=[-20, -20, -10],
            high=[20, 20, 10],
            size=(n_random, 3)
        )
        true_points_3d[n_structured:] = random_points
    
    # Generate observations by projecting points
    observations = []
    for i, camera_pose in enumerate(true_camera_poses):
        # Project all points to this camera
        projected_points = camera_model.project(
            true_points_3d, 
            (camera_pose.rotation, camera_pose.translation)
        )
        
        # Add observations for points that are visible (in front of camera)
        for j, point_2d in enumerate(projected_points):
            # Check if point is in front of camera
            point_cam = camera_pose.rotation @ (true_points_3d[j] - camera_pose.translation)
            if point_cam[2] > 0:  # Point is in front of camera
                # Add Gaussian noise to observation
                noisy_point = point_2d + np.random.normal(0, noise_std, 2)
                observations.append(Observation(i, j, noisy_point))
    
    print(f"Generated {len(observations)} observations")
    
    # Create perturbed initial parameters for optimization
    perturbed_camera_poses, perturbed_points_3d = perturb_parameters(
        true_camera_poses, true_points_3d
    )
    
    # Create bundle adjustment data
    bundle_data = BundleAdjustmentData(
        camera_poses=perturbed_camera_poses,
        points_3d=perturbed_points_3d,
        observations=observations,
        camera_model=camera_model
    )
    
    return bundle_data


def perturb_parameters(
    camera_poses: List[CameraPose],
    points_3d: npt.NDArray[np.float64],
    rotation_perturbation: float = 0.15,
    translation_perturbation: float = 1.5,
    point_perturbation: float = 3.0
) -> Tuple[List[CameraPose], npt.NDArray[np.float64]]:
    """
    Add perturbations to camera poses and 3D points for realistic optimization testing.
    
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


def load_colmap_dataset(images_txt: str, points3D_txt: str) -> BundleAdjustmentData:
    """
    Load COLMAP dataset with reasonable default camera model.
    
    Args:
        images_txt: Path to COLMAP images.txt file
        points3D_txt: Path to COLMAP points3D.txt file
        
    Returns:
        BundleAdjustmentData object with loaded COLMAP data
    """
    print(f"Loading COLMAP dataset:")
    print(f"  Images: {images_txt}")
    print(f"  Points3D: {points3D_txt}")
    
    # Create a reasonable camera model for COLMAP data
    # COLMAP typically outputs focal length in pixels
    # Principal point is often (0,0) or image center
    focal_length = 1000.0  # Default focal length in pixels
    principal_point = (0.0, 0.0)  # COLMAP often uses (0,0) as principal point
    
    camera_model = CameraModel(focal_length, principal_point)
    
    # Load the bundle data
    bundle_data = load_colmap_bundle(images_txt, points3D_txt, camera_model)
    
    # Print summary and validate
    print_colmap_summary(bundle_data)
    
    # Validate the data
    validation_stats = validate_colmap_data(
        bundle_data.camera_poses,
        bundle_data.points_3d,
        bundle_data.observations
    )
    
    # Print validation results
    print(f"\nValidation Results:")
    print(f"  Cameras: {validation_stats['num_cameras']}")
    print(f"  3D Points: {validation_stats['num_points']}")
    print(f"  Observations: {validation_stats['num_observations']}")
    
    if validation_stats['warnings']:
        print(f"\nWarnings:")
        for warning in validation_stats['warnings']:
            print(f"  ⚠ {warning}")
    
    if validation_stats['errors']:
        print(f"\nErrors:")
        for error in validation_stats['errors']:
            print(f"  ❌ {error}")
        raise ValueError("COLMAP data validation failed")
    
    return bundle_data


def run_bundle_adjustment_demo(dataset_type: str, **kwargs) -> None:
    """
    Main routine demonstrating the bundle adjustment library.
    
    Args:
        dataset_type: Type of dataset ('synthetic' or 'colmap')
        **kwargs: Additional arguments (images_txt, points3D_txt for COLMAP)
    """
    print("=" * 60)
    print(f"Bundle Adjustment Tool - {dataset_type.title()} Dataset")
    print("=" * 60)
    
    # Step 1: Load or create dataset
    print(f"\n1. Loading {dataset_type} dataset...")
    
    if dataset_type == 'synthetic':
        bundle_data = create_synthetic_dataset(
            num_cameras=6,
            num_points=100,
            noise_std=1.0,
            random_seed=42
        )
    elif dataset_type == 'colmap':
        images_txt = kwargs.get('images_txt')
        points3D_txt = kwargs.get('points3D_txt')
        
        if not images_txt or not points3D_txt:
            raise ValueError("COLMAP dataset requires --images_txt and --points3D_txt arguments")
        
        bundle_data = load_colmap_dataset(images_txt, points3D_txt)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Validate the dataset
    bundle_data.validate()
    print(f"✓ Dataset validated: {len(bundle_data.camera_poses)} cameras, "
          f"{bundle_data.points_3d.shape[0]} points, {len(bundle_data.observations)} observations")
    
    # Step 2: Compute initial state
    print("\n2. Computing initial state...")
    initial_camera_poses = bundle_data.camera_poses
    initial_points_3d = bundle_data.points_3d
    
    initial_residuals = compute_residuals(bundle_data, initial_camera_poses, initial_points_3d)
    initial_residual_norm = compute_reprojection_error(bundle_data, initial_camera_poses, initial_points_3d)
    
    print(f"Initial residual norm: {initial_residual_norm:.6f}")
    
    # Step 3: Visualize initial state
    print("\n3. Visualizing initial state...")
    plot_reprojection_errors(initial_residuals, f"Initial Reprojection Errors - {dataset_type.title()}")
    plot_cameras_and_points(initial_camera_poses, initial_points_3d, f"Initial 3D Scene - {dataset_type.title()}")
    
    # Step 4: Run bundle adjustment optimization
    print("\n4. Running bundle adjustment optimization...")
    solver = SparseLMSolver(
        data=bundle_data,
        max_iterations=25,
        initial_damping=1.0,
        damping_factor=10.0,
        convergence_threshold=1e-6
    )
    
    optimized_camera_poses, optimized_points_3d, final_residual_norm = solver.run()
    
    # Step 5: Compute final state
    print("\n5. Computing final state...")
    final_residuals = compute_residuals(bundle_data, optimized_camera_poses, optimized_points_3d)
    
    # Step 6: Visualize final state
    print("\n6. Visualizing final state...")
    plot_reprojection_errors(final_residuals, f"Final Reprojection Errors - {dataset_type.title()}")
    plot_cameras_and_points(optimized_camera_poses, optimized_points_3d, f"Optimized 3D Scene - {dataset_type.title()}")
    
    # Step 7: Print comprehensive summary
    print_summary(
        initial_residuals, final_residuals,
        initial_camera_poses, optimized_camera_poses,
        initial_points_3d, optimized_points_3d,
        initial_residual_norm, final_residual_norm,
        dataset_type
    )


def print_summary(
    initial_residuals: npt.NDArray[np.float64],
    final_residuals: npt.NDArray[np.float64],
    initial_camera_poses: List[CameraPose],
    optimized_camera_poses: List[CameraPose],
    initial_points_3d: npt.NDArray[np.float64],
    optimized_points_3d: npt.NDArray[np.float64],
    initial_residual_norm: float,
    final_residual_norm: float,
    dataset_type: str
) -> None:
    """
    Print comprehensive summary of bundle adjustment results.
    
    Args:
        initial_residuals: Initial reprojection residuals
        final_residuals: Final reprojection residuals
        initial_camera_poses: Initial camera poses
        optimized_camera_poses: Optimized camera poses
        initial_points_3d: Initial 3D points
        optimized_points_3d: Optimized 3D points
        initial_residual_norm: Initial residual norm
        final_residual_norm: Final residual norm
        dataset_type: Type of dataset used
    """
    print("\n" + "=" * 60)
    print(f"Bundle Adjustment Results Summary - {dataset_type.title()} Dataset")
    print("=" * 60)
    
    # Residual analysis
    print("\n📊 Residual Analysis:")
    print(f"  Initial residual norm: {initial_residual_norm:.6f}")
    print(f"  Final residual norm: {final_residual_norm:.6f}")
    print(f"  Improvement: {initial_residual_norm - final_residual_norm:.6f}")
    print(f"  Improvement percentage: {((initial_residual_norm - final_residual_norm) / initial_residual_norm * 100):.2f}%")
    
    # Reprojection error statistics
    initial_errors = np.sqrt(initial_residuals[::2]**2 + initial_residuals[1::2]**2)
    final_errors = np.sqrt(final_residuals[::2]**2 + final_residuals[1::2]**2)
    
    print(f"\n📈 Reprojection Error Statistics:")
    print(f"  Initial - Mean: {np.mean(initial_errors):.3f} px, Std: {np.std(initial_errors):.3f} px, Max: {np.max(initial_errors):.3f} px")
    print(f"  Final   - Mean: {np.mean(final_errors):.3f} px, Std: {np.std(final_errors):.3f} px, Max: {np.max(final_errors):.3f} px")
    
    # Camera pose analysis
    print(f"\n📷 Camera Pose Analysis:")
    camera_shifts = []
    for i, (init_pose, opt_pose) in enumerate(zip(initial_camera_poses, optimized_camera_poses)):
        translation_shift = np.linalg.norm(init_pose.translation - opt_pose.translation)
        camera_shifts.append(translation_shift)
        print(f"  Camera {i}: Translation shift = {translation_shift:.3f} units")
    
    print(f"  Average camera shift: {np.mean(camera_shifts):.3f} units")
    print(f"  Max camera shift: {np.max(camera_shifts):.3f} units")
    
    # 3D point analysis
    print(f"\n🎯 3D Point Analysis:")
    point_shifts = np.linalg.norm(optimized_points_3d - initial_points_3d, axis=1)
    print(f"  Average point shift: {np.mean(point_shifts):.3f} units")
    print(f"  Max point shift: {np.max(point_shifts):.3f} units")
    print(f"  Point shift std: {np.std(point_shifts):.3f} units")
    
    # Convergence assessment
    print(f"\n✅ Convergence Assessment:")
    if final_residual_norm < 2.0:
        print("  ✓ Optimization converged successfully")
    else:
        print("  ⚠ Optimization may need more iterations")
    
    if final_residual_norm < initial_residual_norm:
        print("  ✓ Residual norm decreased (optimization effective)")
    else:
        print("  ⚠ Residual norm did not decrease")
    
    if np.mean(final_errors) < np.mean(initial_errors):
        print("  ✓ Average reprojection error improved")
    else:
        print("  ⚠ Average reprojection error did not improve")
    
    print("\n" + "=" * 60)
    print(f"Demonstration completed successfully! 🎉")
    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Bundle Adjustment Tool - Run optimization on synthetic or COLMAP data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data (default)
  python main.py --dataset synthetic
  
  # Run with COLMAP data
  python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['synthetic', 'colmap'],
        default='synthetic',
        help='Type of dataset to use (default: synthetic)'
    )
    
    parser.add_argument(
        '--images_txt',
        type=str,
        help='Path to COLMAP images.txt file (required for colmap dataset)'
    )
    
    parser.add_argument(
        '--points3D_txt',
        type=str,
        help='Path to COLMAP points3D.txt file (required for colmap dataset)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        ValueError: If arguments are invalid
    """
    if args.dataset == 'colmap':
        if not args.images_txt:
            raise ValueError("COLMAP dataset requires --images_txt argument")
        if not args.points3D_txt:
            raise ValueError("COLMAP dataset requires --points3D_txt argument")
        
        # Check if files exist
        if not Path(args.images_txt).exists():
            raise ValueError(f"COLMAP images.txt file not found: {args.images_txt}")
        if not Path(args.points3D_txt).exists():
            raise ValueError(f"COLMAP points3D.txt file not found: {args.points3D_txt}")


def main() -> None:
    """
    Main entry point for the bundle adjustment demonstration.
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Prepare kwargs for dataset loading
        kwargs = {}
        if args.dataset == 'colmap':
            kwargs['images_txt'] = args.images_txt
            kwargs['points3D_txt'] = args.points3D_txt
        
        # Run the demonstration
        run_bundle_adjustment_demo(args.dataset, **kwargs)
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main() 