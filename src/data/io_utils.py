import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation

from .observations import CameraPose, Observation, BundleAdjustmentData
from .camera_models import CameraModel


def load_colmap_cameras(path_to_images_txt: str) -> Tuple[List[CameraPose], List[Observation]]:
    """
    Load camera poses and observations from COLMAP images.txt file.
    
    COLMAP images.txt format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    
    Args:
        path_to_images_txt: Path to COLMAP images.txt file
        
    Returns:
        Tuple of (camera_poses, observations) where:
        - camera_poses: List of CameraPose objects
        - observations: List of Observation objects
        
    Raises:
        FileNotFoundError: If images.txt file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path_to_images_txt)
    if not path.exists():
        raise FileNotFoundError(f"COLMAP images.txt file not found: {path_to_images_txt}")
    
    camera_poses = []
    observations = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip header comments
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    i = 0
    while i < len(data_lines):
        # Parse camera pose line
        pose_line = data_lines[i]
        pose_parts = pose_line.split()
        
        if len(pose_parts) < 9:
            raise ValueError(f"Invalid camera pose line: {pose_line}")
        
        # Extract camera pose data
        image_id = int(pose_parts[0])
        qw, qx, qy, qz = map(float, pose_parts[1:5])  # Quaternion
        tx, ty, tz = map(float, pose_parts[5:8])      # Translation
        camera_id = int(pose_parts[8])
        
        # Convert quaternion to rotation matrix
        quaternion = np.array([qw, qx, qy, qz])
        rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Create camera pose
        camera_pose = CameraPose(rotation_matrix, np.array([tx, ty, tz]))
        camera_poses.append(camera_pose)
        
        # Parse observations line
        if i + 1 < len(data_lines):
            obs_line = data_lines[i + 1]
            obs_parts = obs_line.split()
            
            # Observations come in groups of 3: (X, Y, POINT3D_ID)
            for j in range(0, len(obs_parts), 3):
                if j + 2 < len(obs_parts):
                    x, y = map(float, obs_parts[j:j+2])
                    point3d_id = int(obs_parts[j+2])
                    
                    # Only add observations with valid 3D points (point3d_id != -1)
                    if point3d_id != -1:
                        observation = Observation(
                            camera_index=len(camera_poses) - 1,  # 0-based index
                            point_index=point3d_id,  # Will be mapped later
                            image_point=np.array([x, y])
                        )
                        observations.append(observation)
        
        i += 2  # Skip to next camera (2 lines per camera)
    
    return camera_poses, observations


def load_colmap_points3D(path_to_points3D_txt: str) -> npt.NDArray[np.float64]:
    """
    Load 3D points from COLMAP points3D.txt file.
    
    COLMAP points3D.txt format:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    
    Args:
        path_to_points3D_txt: Path to COLMAP points3D.txt file
        
    Returns:
        3D points array of shape (N, 3) where N is the number of points
        
    Raises:
        FileNotFoundError: If points3D.txt file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path_to_points3D_txt)
    if not path.exists():
        raise FileNotFoundError(f"COLMAP points3D.txt file not found: {path_to_points3D_txt}")
    
    points_3d = []
    point_id_to_index = {}  # Map COLMAP point IDs to array indices
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip header comments
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    for i, line in enumerate(data_lines):
        parts = line.split()
        
        if len(parts) < 4:
            raise ValueError(f"Invalid 3D point line: {line}")
        
        # Extract point data
        point3d_id = int(parts[0])
        x, y, z = map(float, parts[1:4])
        
        # Store point and create mapping
        points_3d.append([x, y, z])
        point_id_to_index[point3d_id] = i
    
    return np.array(points_3d, dtype=np.float64), point_id_to_index


def load_colmap_bundle(
    path_to_images_txt: str,
    path_to_points3D_txt: str,
    camera_model: CameraModel
) -> BundleAdjustmentData:
    """
    Load complete bundle adjustment data from COLMAP text exports.
    
    Args:
        path_to_images_txt: Path to COLMAP images.txt file
        path_to_points3D_txt: Path to COLMAP points3D.txt file
        camera_model: CameraModel instance with intrinsics
        
    Returns:
        BundleAdjustmentData object with loaded camera poses, 3D points, and observations
        
    Raises:
        FileNotFoundError: If either file doesn't exist
        ValueError: If data is inconsistent or invalid
    """
    # Load camera poses and observations
    camera_poses, observations = load_colmap_cameras(path_to_images_txt)
    
    # Load 3D points
    points_3d, point_id_to_index = load_colmap_points3D(path_to_points3D_txt)
    
    # Validate data consistency
    if len(camera_poses) == 0:
        raise ValueError("No camera poses loaded from images.txt")
    
    if len(points_3d) == 0:
        raise ValueError("No 3D points loaded from points3D.txt")
    
    # Remap observation point indices to match the loaded 3D points
    valid_observations = []
    for obs in observations:
        if obs.point_index in point_id_to_index:
            # Update point index to match the loaded 3D points array
            new_obs = Observation(
                camera_index=obs.camera_index,
                point_index=point_id_to_index[obs.point_index],
                image_point=obs.image_point
            )
            valid_observations.append(new_obs)
        else:
            # Skip observations with invalid point IDs
            continue
    
    if len(valid_observations) == 0:
        raise ValueError("No valid observations found after point index remapping")
    
    # Create bundle adjustment data
    bundle_data = BundleAdjustmentData(
        camera_poses=camera_poses,
        points_3d=points_3d,
        observations=valid_observations,
        camera_model=camera_model
    )
    
    return bundle_data


def validate_colmap_data(
    camera_poses: List[CameraPose],
    points_3d: npt.NDArray[np.float64],
    observations: List[Observation]
) -> Dict[str, any]:
    """
    Validate COLMAP data for consistency and quality.
    
    Args:
        camera_poses: List of camera poses
        points_3d: 3D points array
        observations: List of observations
        
    Returns:
        Dictionary with validation statistics and warnings
    """
    stats = {
        'num_cameras': len(camera_poses),
        'num_points': len(points_3d),
        'num_observations': len(observations),
        'warnings': [],
        'errors': []
    }
    
    # Check for valid camera poses
    for i, pose in enumerate(camera_poses):
        # Check rotation matrix orthogonality
        R = pose.rotation
        orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
        if orthogonality_error > 1e-6:
            stats['warnings'].append(f"Camera {i}: Rotation matrix not orthogonal (error: {orthogonality_error:.6f})")
        
        # Check rotation matrix determinant
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-6:
            stats['warnings'].append(f"Camera {i}: Rotation matrix determinant not 1 (det: {det:.6f})")
    
    # Check observation indices
    camera_indices = set(obs.camera_index for obs in observations)
    point_indices = set(obs.point_index for obs in observations)
    
    if max(camera_indices) >= len(camera_poses):
        stats['errors'].append(f"Invalid camera index: {max(camera_indices)} >= {len(camera_poses)}")
    
    if max(point_indices) >= len(points_3d):
        stats['errors'].append(f"Invalid point index: {max(point_indices)} >= {len(points_3d)}")
    
    # Check for cameras with no observations
    cameras_with_obs = set(obs.camera_index for obs in observations)
    cameras_without_obs = set(range(len(camera_poses))) - cameras_with_obs
    if cameras_without_obs:
        stats['warnings'].append(f"Cameras without observations: {cameras_without_obs}")
    
    # Check for points with no observations
    points_with_obs = set(obs.point_index for obs in observations)
    points_without_obs = set(range(len(points_3d))) - points_with_obs
    if points_without_obs:
        stats['warnings'].append(f"Points without observations: {len(points_without_obs)}")
    
    # Compute observation statistics
    if observations:
        image_coords = np.array([obs.image_point for obs in observations])
        stats['mean_image_coords'] = np.mean(image_coords, axis=0)
        stats['std_image_coords'] = np.std(image_coords, axis=0)
        stats['min_image_coords'] = np.min(image_coords, axis=0)
        stats['max_image_coords'] = np.max(image_coords, axis=0)
    
    return stats


def print_colmap_summary(bundle_data: BundleAdjustmentData) -> None:
    """
    Print a summary of loaded COLMAP data.
    
    Args:
        bundle_data: BundleAdjustmentData object loaded from COLMAP
    """
    print("=" * 60)
    print("COLMAP Dataset Summary")
    print("=" * 60)
    
    print(f"Cameras: {len(bundle_data.camera_poses)}")
    print(f"3D Points: {bundle_data.points_3d.shape[0]}")
    print(f"Observations: {len(bundle_data.observations)}")
    
    # Camera statistics
    translations = np.array([pose.translation for pose in bundle_data.camera_poses])
    print(f"\nCamera Statistics:")
    print(f"  Translation range: X[{translations[:, 0].min():.2f}, {translations[:, 0].max():.2f}]")
    print(f"                    Y[{translations[:, 1].min():.2f}, {translations[:, 1].max():.2f}]")
    print(f"                    Z[{translations[:, 2].min():.2f}, {translations[:, 2].max():.2f}]")
    
    # Point statistics
    print(f"\n3D Point Statistics:")
    print(f"  Position range: X[{bundle_data.points_3d[:, 0].min():.2f}, {bundle_data.points_3d[:, 0].max():.2f}]")
    print(f"                  Y[{bundle_data.points_3d[:, 1].min():.2f}, {bundle_data.points_3d[:, 1].max():.2f}]")
    print(f"                  Z[{bundle_data.points_3d[:, 2].min():.2f}, {bundle_data.points_3d[:, 2].max():.2f}]")
    
    # Observation statistics
    image_coords = np.array([obs.image_point for obs in bundle_data.observations])
    print(f"\nObservation Statistics:")
    print(f"  Image coordinates: X[{image_coords[:, 0].min():.1f}, {image_coords[:, 0].max():.1f}]")
    print(f"                    Y[{image_coords[:, 1].min():.1f}, {image_coords[:, 1].max():.1f}]")
    
    # Camera model info
    print(f"\nCamera Model:")
    print(f"  Focal length: {bundle_data.camera_model.focal_length}")
    print(f"  Principal point: {bundle_data.camera_model.principal_point}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage and testing
    print("COLMAP IO Utilities - Example Usage")
    print("This module provides utilities to load COLMAP text exports.")
    print("To use:")
    print("1. Export your COLMAP reconstruction as text files")
    print("2. Use load_colmap_bundle() to load the data")
    print("3. Validate the data with validate_colmap_data()")
    print("4. Run bundle adjustment on the loaded data") 