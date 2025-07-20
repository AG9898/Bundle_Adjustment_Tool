import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation

from ..data.observations import BundleAdjustmentData, CameraPose
from ..core.residuals import compute_residuals, compute_jacobians, compute_reprojection_error
from .schur_complement import build_normal_equations, solve_schur


class SparseLMSolver:
    """
    Sparse Levenberg-Marquardt solver for bundle adjustment.
    
    Implements the LM algorithm with Schur complement for efficient
    solution of large-scale bundle adjustment problems.
    """
    
    def __init__(
        self,
        data: BundleAdjustmentData,
        max_iterations: int = 20,
        initial_damping: float = 1.0,
        damping_factor: float = 10.0,
        convergence_threshold: float = 1e-6
    ) -> None:
        """
        Initialize the sparse LM solver.
        
        Args:
            data: Bundle adjustment data
            max_iterations: Maximum number of iterations
            initial_damping: Initial Levenberg-Marquardt damping parameter
            damping_factor: Factor to scale damping parameter
            convergence_threshold: Threshold for convergence checking
        """
        self.data = data
        self.max_iterations = max_iterations
        self.initial_damping = initial_damping
        self.damping_factor = damping_factor
        self.convergence_threshold = convergence_threshold
        
        # Initialize damping parameter
        self.damping = initial_damping
        
        # Validate data
        self.data.validate()
    
    def run(self) -> Tuple[List[CameraPose], npt.NDArray[np.float64], float]:
        """
        Run the sparse Levenberg-Marquardt optimization.
        
        Returns:
            Tuple of (optimized_camera_poses, optimized_points_3d, final_residual_norm)
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
        
        print(f"Iteration 0: Residual norm = {current_residual_norm:.6f}")
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\nIteration {iteration}:")
            print(f"  Damping parameter: {self.damping:.6f}")
            
            # Step 1: Compute residuals and Jacobians
            residuals = compute_residuals(self.data, camera_poses, points_3d)
            J_cam, J_points = compute_jacobians(self.data, camera_poses, points_3d)
            
            # Step 2: Build normal equations
            A, B, C, rhs_cam, rhs_points = build_normal_equations(
                J_cam, J_points, residuals, self.damping
            )
            
            # Step 3: Solve using Schur complement
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
            
            # Step 7: Check convergence
            if self._check_convergence(delta_cam, delta_points):
                print(f"\nConvergence reached at iteration {iteration}")
                break
        
        print(f"\nOptimization completed:")
        print(f"  Final residual norm: {current_residual_norm:.6f}")
        print(f"  Improvement: {initial_residual_norm - current_residual_norm:.6f}")
        
        return camera_poses, points_3d, current_residual_norm
    
    def _update_camera_params(
        self, 
        camera_poses: List[CameraPose], 
        delta_cam: npt.NDArray[np.float64]
    ) -> List[CameraPose]:
        """
        Update camera parameters using delta updates.
        
        Args:
            camera_poses: Current camera poses
            delta_cam: Camera parameter updates (6*num_cameras,)
            
        Returns:
            Updated camera poses
        """
        num_cameras = len(camera_poses)
        assert len(delta_cam) == 6 * num_cameras, \
            f"Delta camera params length {len(delta_cam)} must be 6 * {num_cameras}"
        
        updated_poses = []
        
        for i in range(num_cameras):
            # Extract updates for this camera
            start_idx = 6 * i
            delta_rot = delta_cam[start_idx:start_idx + 3]
            delta_trans = delta_cam[start_idx + 3:start_idx + 6]
            
            # Update rotation using Rodrigues formula
            current_rot = camera_poses[i].rotation
            delta_rot_matrix = self._rodrigues_to_rotation(delta_rot)
            new_rot = delta_rot_matrix @ current_rot
            
            # Update translation
            new_trans = camera_poses[i].translation + delta_trans
            
            updated_poses.append(CameraPose(new_rot, new_trans))
        
        return updated_poses
    
    def _update_points_3d(
        self, 
        points_3d: npt.NDArray[np.float64], 
        delta_points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Update 3D points using delta updates.
        
        Args:
            points_3d: Current 3D points (Nx3)
            delta_points: Point updates (3*N,)
            
        Returns:
            Updated 3D points (Nx3)
        """
        num_points = points_3d.shape[0]
        assert len(delta_points) == 3 * num_points, \
            f"Delta points length {len(delta_points)} must be 3 * {num_points}"
        
        # Reshape delta_points to (N, 3) and add to current points
        delta_points_reshaped = delta_points.reshape(num_points, 3)
        return points_3d + delta_points_reshaped
    
    def _rodrigues_to_rotation(self, rotation_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Convert rotation vector (Rodrigues) to rotation matrix.
        
        Args:
            rotation_vector: 3D rotation vector
            
        Returns:
            3x3 rotation matrix
        """
        theta = np.linalg.norm(rotation_vector)
        if theta < 1e-8:
            return np.eye(3)
        
        k = rotation_vector / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    def _check_convergence(
        self, 
        delta_cam: npt.NDArray[np.float64], 
        delta_points: npt.NDArray[np.float64]
    ) -> bool:
        """
        Check if the optimization has converged.
        
        Args:
            delta_cam: Camera parameter updates
            delta_points: Point updates
            
        Returns:
            True if converged, False otherwise
        """
        # Check parameter update norms
        cam_update_norm = np.linalg.norm(delta_cam)
        point_update_norm = np.linalg.norm(delta_points)
        
        print(f"  Camera update norm: {cam_update_norm:.6f}")
        print(f"  Point update norm: {point_update_norm:.6f}")
        
        # Convergence if both update norms are small
        return (cam_update_norm < self.convergence_threshold and 
                point_update_norm < self.convergence_threshold) 