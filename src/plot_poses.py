import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def read_colmap_cameras(images_txt_path):
    """Extracts camera centers C = -R^T * t from COLMAP images.txt"""
    centers = []
    with open(images_txt_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
            
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            t = np.array([tx, ty, tz]).reshape(3, 1)
            
            # Quaternion to Rotation Matrix
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            
            C = -R.T @ t
            centers.append(C.flatten())
            
    return np.array(centers)

def get_normalized_trajectory(centers):
    """Zero-centers and scales a trajectory so it can be compared visually."""
    centers = np.array(centers)
    centroid = np.mean(centers, axis=0)
    centered = centers - centroid
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale == 0: scale = 1
    return centered / scale

if __name__ == "__main__":
    # 1. Load COLMAP Centers
    colmap_path = "../Dataset/COLMAP_Workspaces/Truck/sparse/0/images.txt"
    if not os.path.exists(colmap_path):
        print(f"Error: Could not find {colmap_path}. Did you run the colmap model_converter?")
        exit()
        
    colmap_centers = read_colmap_cameras(colmap_path)
    
    # 2. Load Custom Centers
    custom_pkl_path = "custom_sfm_map.pkl"
    if not os.path.exists(custom_pkl_path):
        print(f"Error: Could not find {custom_pkl_path}.")
        exit()
        
    with open(custom_pkl_path, "rb") as f:
        custom_data = pickle.load(f)
    
    # Extract R and t from your custom cameras using the correct key
    custom_poses = custom_data.get('camera_poses', [])
    
    # If it's a dictionary (e.g., {img_idx: pose}), get the values
    if isinstance(custom_poses, dict):
        custom_poses = custom_poses.values()
        
    custom_centers = []
    for cam in custom_poses:
        # Handle different ways the custom pipeline might have saved the pose
        if isinstance(cam, dict):
            R = cam['R']
            t = cam['t']
        elif isinstance(cam, tuple) or isinstance(cam, list):
            R = cam[0]
            t = cam[1]
        else:
            R = cam.R
            t = cam.t
            
        t = t.reshape(3, 1)
        C = -R.T @ t
        custom_centers.append(C.flatten())
        
    custom_centers = np.array(custom_centers)

    # 3. Normalize both trajectories to compare their SHAPES
    norm_colmap = get_normalized_trajectory(colmap_centers)
    norm_custom = get_normalized_trajectory(custom_centers)
    
    # 4. Plot them together
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot COLMAP Trajectory (Blue)
    ax.plot(norm_colmap[:, 0], norm_colmap[:, 1], norm_colmap[:, 2], 
            label="COLMAP Trajectory", color='blue', marker='o', markersize=4, linestyle='-')
    
    # Plot Custom Trajectory (Red)
    ax.plot(norm_custom[:, 0], norm_custom[:, 1], norm_custom[:, 2], 
            label="Custom Trajectory (6 Frames)", color='red', marker='x', markersize=6, linestyle='--', linewidth=2)

    ax.set_title("Camera Pose Comparison (Normalized)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()