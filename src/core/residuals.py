import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from scipy.sparse import csr_matrix, lil_matrix, csr_array
from scipy.spatial.transform import Rotation

from ..data.observations import BundleAdjustmentData, CameraPose


def rodrigues_to_rotation(rotation_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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


def rotation_to_rodrigues(rotation_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to rotation vector (Rodrigues).
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        3D rotation vector
    """
    # Use scipy's Rotation for robust conversion
    r = Rotation.from_matrix(rotation_matrix)
    return r.as_rotvec()


def compute_residuals(
    data: BundleAdjustmentData,
    camera_params: List[CameraPose],
    points_3d: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute reprojection residuals for bundle adjustment.
    
    Args:
        data: Bundle adjustment data containing observations and camera model
        camera_params: Updated camera poses
        points_3d: Updated 3D points (Nx3 array)
        
    Returns:
        Residuals vector of shape (2M,) where M is number of observations
    """
    # Validate inputs
    assert len(camera_params) == len(data.camera_poses), \
        f"Number of camera params ({len(camera_params)}) must match data ({len(data.camera_poses)})"
    assert points_3d.shape == data.points_3d.shape, \
        f"Points shape {points_3d.shape} must match data shape {data.points_3d.shape}"
    
    num_observations = len(data.observations)
    residuals = np.zeros(2 * num_observations)
    
    for i, obs in enumerate(data.observations):
        # Get camera pose and 3D point
        camera_pose = camera_params[obs.camera_index]
        point_3d = points_3d[obs.point_index]
        
        # Transform point to camera coordinates: X_cam = R @ (X_world - t)
        point_cam = camera_pose.rotation @ (point_3d - camera_pose.translation)
        
        # Project to 2D using camera model (pass both points_3d and pose)
        point_2d_projected = data.camera_model.project(
            point_3d.reshape(1, 3),
            (camera_pose.rotation, camera_pose.translation)
        )[0]  # Remove batch dimension
        
        # Compute residual: observed - projected
        residual = obs.image_point - point_2d_projected
        residuals[2*i:2*i+2] = residual
    
    return residuals


def compute_jacobians(
    data: BundleAdjustmentData,
    camera_params: List[CameraPose],
    points_3d: npt.NDArray[np.float64]
) -> Tuple[csr_array, csr_array]:
    """
    Compute sparse Jacobian matrices for bundle adjustment.
    
    Args:
        data: Bundle adjustment data
        camera_params: Updated camera poses
        points_3d: Updated 3D points (Nx3 array)
        
    Returns:
        Tuple of (J_cam, J_points) where:
        - J_cam: Sparse Jacobian wrt camera parameters (2M, 6*num_cameras)
        - J_points: Sparse Jacobian wrt 3D points (2M, 3*num_points)
    """
    # Validate inputs
    assert len(camera_params) == len(data.camera_poses), \
        f"Number of camera params ({len(camera_params)}) must match data ({len(data.camera_poses)})"
    assert points_3d.shape == data.points_3d.shape, \
        f"Points shape {points_3d.shape} must match data shape {data.points_3d.shape}"
    
    num_observations = len(data.observations)
    num_cameras = len(camera_params)
    num_points = points_3d.shape[0]
    
    # Initialize sparse matrices using LIL format for efficient construction
    J_cam = lil_matrix((2 * num_observations, 6 * num_cameras))
    J_points = lil_matrix((2 * num_observations, 3 * num_points))
    
    for i, obs in enumerate(data.observations):
        camera_idx = obs.camera_index
        point_idx = obs.point_index
        
        # Get camera pose and 3D point
        camera_pose = camera_params[camera_idx]
        point_3d = points_3d[point_idx]
        
        # Transform point to camera coordinates
        point_cam = camera_pose.rotation @ (point_3d - camera_pose.translation)
        X, Y, Z = point_cam
        
        # Skip if point is behind camera
        if Z <= 0:
            continue
        
        # Compute normalized coordinates
        x_norm = X / Z
        y_norm = Y / Z
        
        # Apply focal length and principal point
        f = data.camera_model.focal_length
        cx, cy = data.camera_model.principal_point
        
        x_proj = f * x_norm + cx
        y_proj = f * y_norm + cy
        
        # Jacobian wrt 3D point (analytical)
        # d(proj)/d(point_3d) = d(proj)/d(point_cam) * d(point_cam)/d(point_3d)
        # d(point_cam)/d(point_3d) = R
        # d(proj)/d(point_cam) = f * [1/Z, 0, -X/Z²; 0, 1/Z, -Y/Z²]
        
        d_proj_d_cam = np.array([
            [f/Z, 0, -f*X/(Z*Z)],
            [0, f/Z, -f*Y/(Z*Z)]
        ])
        
        d_cam_d_point = camera_pose.rotation
        d_proj_d_point = d_proj_d_cam @ d_cam_d_point
        
        # Fill Jacobian wrt 3D points
        J_points[2*i:2*i+2, 3*point_idx:3*point_idx+3] = d_proj_d_point
        
        # Jacobian wrt camera parameters (analytical)
        # For rotation: use small angle approximation around current rotation
        # For translation: d(point_cam)/d(translation) = -R
        
        # Translation Jacobian
        d_cam_d_trans = -camera_pose.rotation
        d_proj_d_trans = d_proj_d_cam @ d_cam_d_trans
        
        # Rotation Jacobian (simplified - using cross product approximation)
        # For small rotations around current pose
        d_cam_d_rot = np.array([
            [0, -Z, Y],
            [Z, 0, -X],
            [-Y, X, 0]
        ])
        d_proj_d_rot = d_proj_d_cam @ d_cam_d_rot
        
        # Fill Jacobian wrt camera parameters
        cam_param_start = 6 * camera_idx
        J_cam[2*i:2*i+2, cam_param_start:cam_param_start+3] = d_proj_d_rot  # Rotation
        J_cam[2*i:2*i+2, cam_param_start+3:cam_param_start+6] = d_proj_d_trans  # Translation
    
    # Convert to CSR format for efficient operations
    return J_cam.tocsr(), J_points.tocsr()


def compute_reprojection_error(
    data: BundleAdjustmentData,
    camera_params: List[CameraPose],
    points_3d: npt.NDArray[np.float64]
) -> float:
    """
    Compute mean reprojection error.
    
    Args:
        data: Bundle adjustment data
        camera_params: Camera poses
        points_3d: 3D points
        
    Returns:
        Mean reprojection error in pixels
    """
    residuals = compute_residuals(data, camera_params, points_3d)
    return np.sqrt(np.mean(residuals**2)) 