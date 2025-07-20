import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional

from ..data.observations import CameraPose


def plot_cameras_and_points(
    camera_poses: List[CameraPose],
    points_3d: npt.NDArray[np.float64],
    title: Optional[str] = None,
    camera_scale: float = 1.0,
    point_size: float = 10.0
) -> None:
    """
    Plot 3D cameras and points with camera orientations.
    
    Args:
        camera_poses: List of camera poses with rotation and translation
        points_3d: 3D points array of shape (N, 3)
        title: Optional title for the plot
        camera_scale: Scale factor for camera frusta size
        point_size: Size of point markers
    """
    # Validate inputs
    assert len(camera_poses) > 0, "At least one camera pose must be provided"
    assert points_3d.ndim == 2 and points_3d.shape[1] == 3, \
        f"points_3d must be Nx3 array, got shape {points_3d.shape}"
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    if len(points_3d) > 0:
        ax.scatter(
            points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
            c='blue', s=point_size, alpha=0.6, label='3D Points'
        )
    
    # Plot cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(camera_poses)))
    
    for i, pose in enumerate(camera_poses):
        # Camera center (translation)
        camera_center = pose.translation
        
        # Plot camera center
        ax.scatter(
            camera_center[0], camera_center[1], camera_center[2],
            c=[colors[i]], s=100, marker='o', edgecolors='black', linewidth=2,
            label=f'Camera {i}' if i < 5 else None  # Limit legend entries
        )
        
        # Plot camera orientation (frustum)
        plot_camera_frustum(ax, pose, camera_scale, colors[i])
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    if title is None:
        title = f"Bundle Adjustment Result\n"
        title += f"Cameras: {len(camera_poses)}, Points: {len(points_3d)}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend (only for first few cameras to avoid clutter)
    if len(camera_poses) <= 5:
        ax.legend(fontsize=10)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def plot_camera_frustum(
    ax: Axes3D,
    pose: CameraPose,
    scale: float = 1.0,
    color: npt.NDArray[np.float64] = np.array([1, 0, 0, 1])
) -> None:
    """
    Plot a simple camera frustum showing camera orientation.
    
    Args:
        ax: 3D axes object
        pose: Camera pose
        scale: Scale factor for frustum size
        color: Color for the frustum
    """
    # Camera center
    center = pose.translation
    
    # Camera forward direction (negative Z in camera coordinates)
    forward = -pose.rotation[:, 2]  # Third column of rotation matrix
    
    # Camera up direction (Y in camera coordinates)
    up = pose.rotation[:, 1]  # Second column of rotation matrix
    
    # Camera right direction (X in camera coordinates)
    right = pose.rotation[:, 0]  # First column of rotation matrix
    
    # Frustum parameters
    frustum_length = 0.5 * scale
    frustum_width = 0.3 * scale
    frustum_height = 0.2 * scale
    
    # Frustum corners in camera coordinates
    corners_cam = np.array([
        [-frustum_width/2, -frustum_height/2, frustum_length],  # Bottom-left
        [frustum_width/2, -frustum_height/2, frustum_length],   # Bottom-right
        [frustum_width/2, frustum_height/2, frustum_length],    # Top-right
        [-frustum_width/2, frustum_height/2, frustum_length],   # Top-left
        [0, 0, 0]  # Camera center
    ])
    
    # Transform corners to world coordinates
    corners_world = np.zeros_like(corners_cam)
    for i in range(len(corners_cam)):
        corners_world[i] = center + pose.rotation @ corners_cam[i]
    
    # Plot frustum edges
    # Base rectangle
    for i in range(4):
        start = corners_world[i]
        end = corners_world[(i + 1) % 4]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                color=color, linewidth=2)
    
    # Lines from base to camera center
    for i in range(4):
        start = corners_world[i]
        end = corners_world[4]  # Camera center
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                color=color, linewidth=1, alpha=0.7)


def plot_camera_trajectory(
    camera_poses: List[CameraPose],
    title: Optional[str] = None,
    show_orientations: bool = True
) -> None:
    """
    Plot camera trajectory showing camera centers and orientations over time.
    
    Args:
        camera_poses: List of camera poses in chronological order
        title: Optional title for the plot
        show_orientations: Whether to show camera orientations
    """
    if len(camera_poses) < 2:
        print("Warning: Need at least 2 camera poses for trajectory plot")
        return
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera centers
    centers = np.array([pose.translation for pose in camera_poses])
    
    # Plot trajectory line
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 
            'b-', linewidth=2, alpha=0.7, label='Camera Trajectory')
    
    # Plot camera positions
    colors = plt.cm.viridis(np.linspace(0, 1, len(camera_poses)))
    
    for i, (pose, color) in enumerate(zip(camera_poses, colors)):
        # Plot camera center
        ax.scatter(
            pose.translation[0], pose.translation[1], pose.translation[2],
            c=[color], s=50, marker='o', edgecolors='black', linewidth=1
        )
        
        # Plot camera orientation if requested
        if show_orientations:
            plot_camera_frustum(ax, pose, scale=0.3, color=color)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    if title is None:
        title = f"Camera Trajectory\n{len(camera_poses)} Cameras"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing camera and points plotting...")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # Create synthetic camera poses
    camera_poses = []
    for i in range(5):
        # Random rotation
        angle = i * np.pi / 4
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Random translation
        translation = np.array([i * 2, np.sin(i), np.cos(i)])
        
        camera_poses.append(CameraPose(rotation, translation))
    
    # Create synthetic 3D points
    points_3d = np.random.randn(100, 3) * 5
    
    # Plot cameras and points
    plot_cameras_and_points(camera_poses, points_3d, "Synthetic Test Data")
    
    # Plot camera trajectory
    plot_camera_trajectory(camera_poses, "Synthetic Camera Trajectory") 