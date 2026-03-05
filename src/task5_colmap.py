import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Import both functions so we have raw data for the report and KD-Tree for Task 6
from utils.colmap_parsers import build_kd_tree_and_lookup, read_points3d_binary

if __name__ == "__main__":
    db_path = "../Dataset/COLMAP_Workspaces/Truck/database.db"
    pts_path = "../Dataset/COLMAP_Workspaces/Truck/sparse/0/points3D.bin"
    gt_path = "../Dataset/GT_ply_files/Truck.ply"
    
    if not os.path.exists(pts_path):
        print(f"Error: Could not find COLMAP output at {pts_path}. Did it finish running?")
    else:
        # ==========================================
        # 1. EXTRACT DATA & SAVE MAP FOR TASK 6
        # ==========================================
        print("Loading raw points dict to get Reprojection Errors and Density...")
        points3D_dict = read_points3d_binary(pts_path)
        
        print("\nBuilding the KD-Tree for localization...")
        flann_matcher, points_3d, descriptors = build_kd_tree_and_lookup(db_path, pts_path)
        
        map_data = {
            "points_3d": points_3d,
            "descriptors": descriptors
        }
        
        save_path = "colmap_truck_map.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(map_data, f)
            
        print(f"\nMap data saved to {save_path}")
        
        # ==========================================
        # 2. TASK 5 REPORT DELIVERABLES (TRUCK)
        # ==========================================
        print("\n" + "="*40)
        print("TASK 5 DELIVERABLES COMPARISON")
        print("="*40)

        # Reprojection Error & Density
        colmap_errors = [pt.error for pt in points3D_dict.values()]
        mean_repr_err = np.mean(colmap_errors)
        density = len(points3D_dict)

        print("\n--- (b) Reprojection Error ---")
        print(f"COLMAP Pipeline: {mean_repr_err:.4f} px")

        print("\n--- (c) Reconstruction Quality ---")
        print(f"COLMAP Sparse Map Density: {density} points")

        # ==========================================
        # 3. CHAMFER DISTANCE & VISUALIZATION
        # ==========================================
        if os.path.exists(gt_path):
            print("\nCalculating Chamfer Distance...")
            gt_pcd = o3d.io.read_point_cloud(gt_path)
            gt_points = np.asarray(gt_pcd.points)
                
            def get_normalized_points(pts):
                centered = pts - np.mean(pts, axis=0)
                scale = np.mean(np.linalg.norm(centered, axis=1))
                return centered / scale

            norm_colmap = get_normalized_points(points_3d)
            norm_gt = get_normalized_points(gt_points)

            tree_colmap = KDTree(norm_colmap)
            tree_gt = KDTree(norm_gt)
                
            d_colmap_to_gt, _ = tree_gt.query(norm_colmap)
            d_gt_to_colmap, _ = tree_colmap.query(norm_gt)
                
            chamfer_dist = (np.mean(d_colmap_to_gt**2) + np.mean(d_gt_to_colmap**2)) / 2
            print(f"COLMAP Chamfer Distance: {chamfer_dist:.8f}")

            # Qualitative Visualization 
            print("\nGenerating Qualitative Visualization... (Close the plot window to exit)")
            fig = plt.figure(figsize=(10, 5))
            
            # Plot Ground Truth
            ax1 = fig.add_subplot(121, projection='3d')
            gt_sub = norm_gt[::10] # Subsample GT for faster plotting
            ax1.scatter(gt_sub[:,0], gt_sub[:,1], gt_sub[:,2], s=0.5, c='gray')
            ax1.set_title("Ground Truth (Normalized)")
            
            # Plot COLMAP Reconstruction
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(norm_colmap[:,0], norm_colmap[:,1], norm_colmap[:,2], s=0.5, c='blue')
            ax2.set_title("COLMAP Reconstruction (Normalized)")
            
            plt.show()
        else:
            print(f"\nGround truth file not found at {gt_path} to calculate Chamfer distance.")