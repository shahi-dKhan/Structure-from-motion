import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.features import get_camera_intrinsics, extract_and_match_features, estimate_essential_matrix, draw_epipolar_lines 
from utils.triangulation import decompose_essential_matrix, linear_triangulation, non_linear_refine_pt, disambiguate_poses
from utils.helpers import *
from utils.increment import *
from utils.bundle_adjustment import build_observations, pack_parameters, build_jacobian_sparsity, ba_residuals, run_bundle_adjustment, normalize_cloud, compute_chamfer_distance

import argparse
import pickle
import open3d as o3d

def extract_frames_from_video(video_path, interval=400):
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        
        if count % interval == 0:
            yield frame  # Yield hands the frame back immediately, pausing the loop
            
        count += 1
        
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../Dataset/Split_A/split_a_truck-004.mp4', help='Path to input video file')
    parser.add_argument('--interval', type=int, default=120, help='Frame extraction interval (in frames)')
    args = parser.parse_args()
    video_file = args.video
    print(f"Streaming frames from {video_file} in real-time...")
    frame_stream = extract_frames_from_video(video_file, interval=args.interval)
    print("Loading pre-computed 3D map...")
    
    map_pts_stored = []
    camera_poses_stored = []
    K_stored = None
    if os.path.exists("custom_sfm_map.pkl"):
        with open("custom_sfm_map.pkl", "rb") as f:
            map_data = pickle.load(f)

        map_pts_stored = map_data["map_points"]
        camera_poses_stored = map_data["camera_poses"]
        K_stored = map_data["K"]

        print(f"Loaded {len(map_pts_stored)} 3D points and {len(camera_poses_stored)} cameras.")
    try:
        ######
        #------- TASK 2 ----------
        ######
        if len(camera_poses_stored) > 0 and len(map_pts_stored) > 0:
            print("Pre-computed map found. Skipping initialization and running incremental SfM...")
            map_points = map_pts_stored
            camera_poses = camera_poses_stored
            K = K_stored
        else:
            img1_color = next(frame_stream)
            K = get_camera_intrinsics(img1_color.shape)
            img2_color = next(frame_stream)
            
            print("Matching features...")
            img1_gray, img2_gray, kp1, kp2, desc1, desc2, pts1, pts2, good_matches, matched_desc1, matched_desc2 = extract_and_match_features(img1_color, img2_color)
            
            print("Estimating Essential Matrix...")
            E1, E2, mask = estimate_essential_matrix(pts1, pts2, K)
            mask = mask.astype(bool)
            
            print(f"Total Matches: {len(pts1)}")
            print(f"RANSAC Inliers: {np.sum(mask)}")
            
            K_inv = np.linalg.inv(K)
            F1 = K_inv.T @ E1 @ K_inv
            F2 = K_inv.T @ E2 @ K_inv
            mask = mask.astype(bool)
            pts1_inliers = pts1[mask]
            pts2_inliers = pts2[mask]
            desc1_inliers = matched_desc1[mask]
            desc2_inliers = matched_desc2[mask]
            print(f"Total Matches: {len(pts1)}")
            print(f"RANSAC Inliers: {np.sum(mask)}")
            # draw_epipolar_lines(img1_color, img2_color, pts1, pts2, F1, title="E1: All Correspondences")
            # draw_epipolar_lines(img1_color, img2_color, pts1_inliers, pts2_inliers, F2, title="E2: Robust (RANSAC)")
            
            
            #Task 2
            poses = decompose_essential_matrix(E2)
            best_pose, X_linear = disambiguate_poses(poses, K, pts1_inliers, pts2_inliers)
            R_best, t_best = best_pose
            print("Best Pose (R, t):")
            print("R:\n", R_best)
            print("t:\n", t_best)
            
            P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
            P2 = K @ np.hstack((R_best, t_best.reshape(3,1)))
            X_refined = []
            for Xi, p1, p2 in zip(X_linear, pts1_inliers, pts2_inliers):
                Xi_h = np.append(Xi, 1)
                Xr = non_linear_refine_pt(Xi_h, [P1, P2], [p1, p2])
                X_refined.append(Xr)
            X_refined = np.array(X_refined)
            
            # =========================
            # -------- TASK 3 ---------
            # =========================
            valid_X, valid_d1, valid_d2, valid_p1, valid_p2 = [], [], [], [], []
            for X, d1, d2, p1, p2 in zip(X_refined, desc1_inliers, desc2_inliers, pts1_inliers, pts2_inliers):
                if X[2] > 0 and np.linalg.norm(X) < 1000:  # Check if the point is in front of the camera and within a reasonable distance
                    valid_X.append(X)
                    valid_d1.append(d1)
                    valid_d2.append(d2)
                    valid_p1.append(p1)
                    valid_p2.append(p2)
            print(f"\nFiltered initial points from {len(X_refined)} to {len(valid_X)}")
            print(f"\nInitializing incremental map...")
            map_points, camera_poses = initialize_map(valid_X, valid_d1, valid_d2, R_best, t_best, valid_p1, valid_p2)
            
            
            print("Initial cameras:", len(camera_poses))
            print("Initial 3D points:", len(map_points))
            registered_frames = [(0, kp1, desc1), (1, kp2, desc2)]

            map_points, camera_poses = run_incremental_sfm(frame_stream,K,map_points,camera_poses,registered_frames=registered_frames,max_frames=6)
        
        # =========================
        # ------ FINAL OUTPUT -----
        # =========================

        print("\nFinal Cameras:", len(camera_poses))
        print("Final 3D Points:", len(map_points))
        X_all = np.array([mp.X for mp in map_points])
        norms = np.linalg.norm(X_all, axis=1)
        print("Median:", np.median(norms))
        print("95 percentile:", np.percentile(norms, 95))
        print("Max:", np.max(norms))
        
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        valid_idx = norms < np.percentile(norms, 95)  # Filter out extreme outliers for better visualization
        X_all = np.array([mp.X for mp in map_points])
        X_vis = X_all[valid_idx]
        ax.scatter(X_vis[:,0], X_vis[:,1], X_vis[:,2], s=3)

        for i, (R, t) in enumerate(camera_poses):
            C = -R.T @ t
            ax.scatter(C[0], C[1], C[2], c='red', s=60)
            ax.text(C[0], C[1], C[2], f'C{i}', size=10, zorder=1, color='k')

        ax.set_title("Incremental Sparse 3D Map")
        plt.show()
        
        # Before BA
        err_before = compute_global_reprojection_error(map_points, camera_poses, K)
        print("Mean reprojection error BEFORE BA:", err_before)

        map_points, camera_poses = run_bundle_adjustment(
           map_points,
           camera_poses,
           K
        )

       # After BA
        err_after = compute_global_reprojection_error(map_points, camera_poses, K)
        print("Mean reprojection error AFTER BA:", err_after)
        
        
    except StopIteration:
        print("Not enough frames extracted to perform matching.")
        
    # print("\nSaving 3D map and camera poses to disk...")
    # map_data = {
    #     "map_points": map_points,
    #     "camera_poses": camera_poses,
    #     "K": K
    # }

    # with open("custom_sfm_map.pkl", "wb") as f:
    #     pickle.dump(map_data, f)
    # print("Saved successfully to 'custom_sfm_map.pkl'.")
    gt_path = "../Dataset/GT_ply_files/Truck.ply"
    gt_mesh = o3d.io.read_point_cloud(gt_path)
    gt_points = np.asarray(gt_mesh.points)
    recon_pts = np.array([mp.X for mp in map_points])
    norm_recon = normalize_cloud(recon_pts)
    norm_gt = normalize_cloud(gt_points)
    chamfer_dist = compute_chamfer_distance(norm_recon, norm_gt)
    print(f"Chamfer Distance to GT: {chamfer_dist:.6f}")
    
    
