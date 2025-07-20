import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from ..data.camera_models import CameraModel
from ..data.observations import BundleAdjustmentData, CameraPose, Observation
from ..solvers.sparse_lm_solver import SparseLMSolver
from ..core.residuals import compute_reprojection_error


class ConvergenceTracker:
    """
    Track convergence history during bundle adjustment optimization.
    """
    
    def __init__(self) -> None:
        self.residual_norms = []
        self.damping_values = []
        self.iterations = []
    
    def add_iteration(self, iteration: int, residual_norm: float, damping: float) -> None:
        """Add iteration data to tracker."""
        self.iterations.append(iteration)
        self.residual_norms.append(residual_norm)
        self.damping_values.append(damping)
    
    def get_convergence_data(self) -> Tuple[List[int], List[float], List[float]]:
        """Get convergence history data."""
        return self.iterations, self.residual_norms, self.damping_values


class TrackingSparseLMSolver(SparseLMSolver):
    """
    Extended SparseLMSolver that tracks convergence history.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = ConvergenceTracker()
    
    def run(self) -> Tuple[List[CameraPose], npt.NDArray[np.float64], float, ConvergenceTracker]:
        """
        Run optimization with convergence tracking.
        
        Returns:
            Tuple of (optimized_camera_poses, optimized_points_3d, final_residual_norm, tracker)
        """
        print(f"Starting sparse LM optimization with {len(self.data.camera_poses)} cameras, "
              f"{self.data.points_3d.shape[0]} points, {len(self.data.observations)} observations")
        
        # Initialize parameters
        camera_poses = [CameraPose(pose.rotation.copy(), pose.translation.copy()) 
                       for pose in self.data.camera_poses]
        points_3d = self.data.points_3d.copy()
        
        # Compute initial residual norm
        initial_residual_norm = compute_reprojection_error(self.data, camera_poses, points_3d)
        current_residual_norm = initial_residual_norm
        
        # Track initial state
        self.tracker.add_iteration(0, current_residual_norm, self.damping)
        
        print(f"Iteration 0: Residual norm = {current_residual_norm:.6f}")
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\nIteration {iteration}:")
            print(f"  Damping parameter: {self.damping:.6f}")
            
            # Step 1: Compute residuals and Jacobians
            from ..core.residuals import compute_residuals, compute_jacobians
            residuals = compute_residuals(self.data, camera_poses, points_3d)
            J_cam, J_points = compute_jacobians(self.data, camera_poses, points_3d)
            
            # Step 2: Build normal equations
            from .schur_complement import build_normal_equations
            A, B, C, rhs_cam, rhs_points = build_normal_equations(
                J_cam, J_points, residuals, self.damping
            )
            
            # Step 3: Solve using Schur complement
            from .schur_complement import solve_schur
            try:
                delta_cam, delta_points = solve_schur(A, B, C, rhs_cam, rhs_points)
            except Exception as e:
                print(f"  Warning: Linear solve failed: {e}")
                # Increase damping and continue
                self.damping *= self.damping_factor
                continue
            
            # Step 4: Apply parameter updates
            camera_poses_new = self._update_camera_params(camera_poses, delta_cam)
            points_3d_new = self._update_points_3d(points_3d, delta_points)
            
            # Step 5: Evaluate new residual norm
            new_residual_norm = compute_reprojection_error(self.data, camera_poses_new, points_3d_new)
            
            print(f"  Current residual: {current_residual_norm:.6f}")
            print(f"  New residual: {new_residual_norm:.6f}")
            
            # Step 6: Check if update improves the solution
            if new_residual_norm < current_residual_norm:
                # Accept the update
                camera_poses = camera_poses_new
                points_3d = points_3d_new
                current_residual_norm = new_residual_norm
                
                # Decrease damping (more like Gauss-Newton)
                self.damping = max(self.damping / self.damping_factor, 1e-8)
                print(f"  Update accepted. Damping decreased to {self.damping:.6f}")
            else:
                # Reject the update, increase damping (more like steepest descent)
                self.damping *= self.damping_factor
                print(f"  Update rejected. Damping increased to {self.damping:.6f}")
            
            # Track iteration
            self.tracker.add_iteration(iteration, current_residual_norm, self.damping)
            
            # Step 7: Check convergence
            if self._check_convergence(delta_cam, delta_points):
                print(f"\nConvergence reached at iteration {iteration}")
                break
        
        print(f"\nOptimization completed:")
        print(f"  Final residual norm: {current_residual_norm:.6f}")
        print(f"  Improvement: {initial_residual_norm - current_residual_norm:.6f}")
        
        return camera_poses, points_3d, current_residual_norm, self.tracker


def create_small_synthetic_scene(
    num_cameras: int = 3,
    num_points: int = 20,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> Tuple[BundleAdjustmentData, List[CameraPose], npt.NDArray[np.float64]]:
    """
    Create a small synthetic scene for convergence testing.
    
    Args:
        num_cameras: Number of cameras
        num_points: Number of 3D points
        noise_std: Standard deviation of observation noise
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (bundle_data, true_camera_poses, true_points_3d)
    """
    np.random.seed(random_seed)
    
    # Create camera model
    focal_length = 1000.0
    principal_point = (640.0, 480.0)
    camera_model = CameraModel(focal_length, principal_point)
    
    # Create true camera poses in a triangle pattern
    true_camera_poses = []
    for i in range(num_cameras):
        angle = i * 2 * np.pi / num_cameras
        radius = 10.0
        
        # Translation: triangle pattern
        translation = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            5.0
        ])
        
        # Rotation: look towards center
        center_direction = np.array([0, 0, 0]) - translation
        center_direction = center_direction / np.linalg.norm(center_direction)
        
        # Create rotation matrix
        up = np.array([0, 0, 1])
        right = np.cross(center_direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, center_direction)
        
        rotation = np.column_stack([right, up, -center_direction])
        
        true_camera_poses.append(CameraPose(rotation, translation))
    
    # Create true 3D points in a smaller box
    true_points_3d = np.random.uniform(
        low=[-10, -10, -5],
        high=[10, 10, 5],
        size=(num_points, 3)
    )
    
    # Generate observations
    observations = []
    for i, camera_pose in enumerate(true_camera_poses):
        # Project all points to this camera
        projected_points = camera_model.project(true_points_3d, (camera_pose.rotation, camera_pose.translation))
        
        # Add observations for visible points
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


def test_damping_behavior() -> None:
    """
    Test the effect of different initial damping values on convergence.
    """
    print("=== Damping Behavior Test ===")
    
    # Create synthetic scene
    print("Creating small synthetic scene...")
    bundle_data, true_camera_poses, true_points_3d = create_small_synthetic_scene(
        num_cameras=3,
        num_points=20,
        noise_std=0.5,
        random_seed=42
    )
    
    print(f"Created scene with {len(bundle_data.camera_poses)} cameras, "
          f"{bundle_data.points_3d.shape[0]} points, {len(bundle_data.observations)} observations")
    
    # Test different initial damping values
    initial_damping_values = [0.1, 1.0, 10.0, 100.0]
    results: Dict[float, Dict] = {}
    
    for initial_damping in initial_damping_values:
        print(f"\n--- Testing initial damping = {initial_damping} ---")
        
        # Perturb parameters for testing
        from .test_synthetic_dataset import perturb_parameters
        perturbed_camera_poses, perturbed_points_3d = perturb_parameters(
            bundle_data.camera_poses,
            bundle_data.points_3d,
            rotation_perturbation=0.05,
            translation_perturbation=0.5,
            point_perturbation=1.0
        )
        
        # Update bundle data with perturbed parameters
        bundle_data.camera_poses = perturbed_camera_poses
        bundle_data.points_3d = perturbed_points_3d
        
        # Run optimization with tracking
        solver = TrackingSparseLMSolver(
            data=bundle_data,
            max_iterations=30,
            initial_damping=initial_damping,
            damping_factor=10.0,
            convergence_threshold=1e-6
        )
        
        optimized_camera_poses, optimized_points_3d, final_residual_norm, tracker = solver.run()
        
        # Store results
        results[initial_damping] = {
            'final_residual_norm': final_residual_norm,
            'num_iterations': len(tracker.iterations) - 1,  # Exclude iteration 0
            'tracker': tracker,
            'converged': final_residual_norm < 1.0
        }
        
        print(f"Final residual: {final_residual_norm:.6f}")
        print(f"Number of iterations: {results[initial_damping]['num_iterations']}")
    
    # Plot convergence curves
    plot_convergence_curves(results)
    
    # Analyze results
    analyze_damping_sensitivity(results)
    
    # Validation assertions
    print("\n=== Validation ===")
    
    # All runs should achieve reasonable convergence
    for damping, result in results.items():
        assert result['converged'], f"Optimization with damping {damping} did not converge"
        assert result['final_residual_norm'] < 2.0, \
            f"Final residual too high for damping {damping}: {result['final_residual_norm']}"
    
    print("✓ All damping values achieved convergence")
    
    # Lower damping should generally require fewer iterations (unless ill-conditioned)
    damping_0_1_iterations = results[0.1]['num_iterations']
    damping_1_0_iterations = results[1.0]['num_iterations']
    
    # Allow some tolerance for numerical differences
    assert abs(damping_0_1_iterations - damping_1_0_iterations) <= 5, \
        f"Unexpected iteration difference: {damping_0_1_iterations} vs {damping_1_0_iterations}"
    
    print("✓ Damping sensitivity analysis completed")
    
    print("\n=== Test Summary ===")
    for damping, result in results.items():
        print(f"Damping {damping}: {result['num_iterations']} iterations, "
              f"final residual {result['final_residual_norm']:.6f}")
    
    print("All tests passed! ✓")


def plot_convergence_curves(results: Dict[float, Dict]) -> None:
    """
    Plot convergence curves for different damping values.
    
    Args:
        results: Dictionary of results for different damping values
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Residual norm vs iteration
    plt.subplot(1, 3, 1)
    for damping, result in results.items():
        tracker = result['tracker']
        plt.semilogy(tracker.iterations, tracker.residual_norms, 
                    marker='o', label=f'λ₀ = {damping}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Damping parameter vs iteration
    plt.subplot(1, 3, 2)
    for damping, result in results.items():
        tracker = result['tracker']
        plt.semilogy(tracker.iterations, tracker.damping_values, 
                    marker='s', label=f'λ₀ = {damping}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Damping Parameter (λ)')
    plt.title('Damping Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final results comparison
    plt.subplot(1, 3, 3)
    dampings = list(results.keys())
    final_residuals = [results[d]['final_residual_norm'] for d in dampings]
    num_iterations = [results[d]['num_iterations'] for d in dampings]
    
    x = np.arange(len(dampings))
    width = 0.35
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, final_residuals, width, label='Final Residual', alpha=0.7)
    bars2 = ax2.bar(x + width/2, num_iterations, width, label='Iterations', alpha=0.7, color='orange')
    
    ax1.set_xlabel('Initial Damping (λ₀)')
    ax1.set_ylabel('Final Residual Norm', color='blue')
    ax2.set_ylabel('Number of Iterations', color='orange')
    ax1.set_title('Final Results Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{d}' for d in dampings])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def analyze_damping_sensitivity(results: Dict[float, Dict]) -> None:
    """
    Analyze sensitivity to initial damping parameter.
    
    Args:
        results: Dictionary of results for different damping values
    """
    print("\n=== Damping Sensitivity Analysis ===")
    
    # Find best performing damping
    best_damping = min(results.keys(), key=lambda d: results[d]['final_residual_norm'])
    worst_damping = max(results.keys(), key=lambda d: results[d]['final_residual_norm'])
    
    print(f"Best initial damping: {best_damping} (residual: {results[best_damping]['final_residual_norm']:.6f})")
    print(f"Worst initial damping: {worst_damping} (residual: {results[worst_damping]['final_residual_norm']:.6f})")
    
    # Analyze iteration efficiency
    fastest_damping = min(results.keys(), key=lambda d: results[d]['num_iterations'])
    slowest_damping = max(results.keys(), key=lambda d: results[d]['num_iterations'])
    
    print(f"Fastest convergence: λ₀ = {fastest_damping} ({results[fastest_damping]['num_iterations']} iterations)")
    print(f"Slowest convergence: λ₀ = {slowest_damping} ({results[slowest_damping]['num_iterations']} iterations)")
    
    # Compute statistics
    final_residuals = [results[d]['final_residual_norm'] for d in results.keys()]
    iterations = [results[d]['num_iterations'] for d in results.keys()]
    
    print(f"Mean final residual: {np.mean(final_residuals):.6f} ± {np.std(final_residuals):.6f}")
    print(f"Mean iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")


if __name__ == "__main__":
    test_damping_behavior() 