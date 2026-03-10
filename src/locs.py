import cv2
import numpy as np
import argparse
from utils.helpers import extract_frames_from_video
import pickle
import os
import matplotlib.pyplot as plt


def custom_map_localization(args, map_data, max_frames=10):
    results = {
        'poses' : [],
        'errors' : [],
        'inlier_ratios' : [],
        'frame_indices' : []
    }
    camera_poses_b = []
    s_b_loc = args.split_b_video
    map_pts_obj = map_data["map_points"]
    K = map_data["K"]
    desc_split_a = np.array([mp.descriptors[0] for mp in map_pts_obj], dtype=np.float32)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    sift = cv2.SIFT_create()
    frame_stream = extract_frames_from_video(s_b_loc, interval=args.interval)
    for i, img in enumerate(frame_stream):
        if i >= max_frames:
            break
        print("Processing a new frame for localization...")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(img_gray, None)
        matches = flann.knnMatch(desc, desc_split_a, k=2)
        good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        
        print(f"Found {len(good_matches)} good matches.")
        pts3D = np.array([map_pts_obj[m.trainIdx].X for m in good_matches], dtype=np.float32)
        pts2D = np.array([kp[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        
        if len(pts3D) < 6:
            print("Not enough matches for reliable localization. Skipping this frame.")
            continue
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3D, pts2D, K, None)
        if success and inliers is not None and len(inliers) >= 6:  
            inliers = inliers.flatten()
            _, rvec, tvec = cv2.solvePnP(pts3D[inliers], pts2D[inliers], K, None, rvec, tvec, useExtrinsicGuess=True)
            img_pts_back, _ = cv2.projectPoints(pts3D[inliers], rvec, tvec, K, None)
            error = np.linalg.norm(img_pts_back.reshape(-1, 2) - pts2D[inliers], axis=1).mean()
            inlier_ratio = len(inliers) / len(good_matches)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            camera_poses_b.append((R, t))
            print(f"Localized camera pose: R=\n{R}\nt={t}\n)")
            results['poses'].append((R, t))
            results['errors'].append(error)
            results['inlier_ratios'].append(inlier_ratio)
            results['frame_indices'].append(i)
        else:
            print("Localization failed for this frame.")
    return results


def report_metrics(results, map_name):
    errors = np.array(results['errors'])
    ratios = np.array(results['inlier_ratios'])
    indices = np.array(results['frame_indices'])

    print(f"\n--- {map_name} Metrics ---")
    print(f"Mean Reprojection Error: {np.mean(errors):.4f} (std: {np.std(errors):.4f})")
    print(f"Mean Inlier Ratio: {np.mean(ratios):.4f}")

    # Plotting A: Reprojection Error vs Frame
    plt.figure(figsize=(10, 4))
    plt.plot(indices, errors, label='Reprojection Error', color='red')
    plt.title(f"{map_name}: Reprojection Error vs Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Error (px)")
    plt.grid(True)
    plt.show()

    # Plotting B: Inlier Ratio vs Frame
    plt.figure(figsize=(10, 4))
    plt.bar(indices, ratios, color='blue', alpha=0.6)
    plt.title(f"{map_name}: Inlier Ratio Stability")
    plt.xlabel("Frame Index")
    plt.ylabel("Ratio")
    plt.show()

def plot_trajectories(map_data, split_b_results, title="Trajectory Comparison"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot Split A (Mapping) - with safety check
    camera_poses_a = map_data.get("camera_poses", [])
    if len(camera_poses_a) > 0:
        split_a_centers = []
        for pose in camera_poses_a:
            # Handle if pose is (R, t) tuple or object
            R, t = pose[0], pose[1]
            C = -R.T @ t.reshape(3, 1)
            split_a_centers.append(C.flatten())
        
        split_a_centers = np.array(split_a_centers)
        ax.plot(split_a_centers[:,0], split_a_centers[:,1], split_a_centers[:,2], 
                'g-', label='Split A (Mapping)', marker='o', markersize=2)
    else:
        print(f"Warning: No Split A mapping poses found for {title}")

    # 2. Plot Split B (Localization) - with safety check
    if len(split_b_results['poses']) > 0:
        split_b_centers = []
        for R, t in split_b_results['poses']:
            C = -R.T @ t.reshape(3, 1)
            split_b_centers.append(C.flatten())
            
        split_b_centers = np.array(split_b_centers)
        ax.plot(split_b_centers[:,0], split_b_centers[:,1], split_b_centers[:,2], 
                'r--', label='Split B (Localization)', marker='x', markersize=4)
    
    ax.set_title(title)
    ax.legend()
    plt.show()
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_a_video', type=str, default= '../Dataset/Split_A/split_a_truck-004.mp4')
    parser.add_argument('--split_b_video', type=str, default= '../Dataset/Split_B/split_b_truck.mp4')
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--stored_descriptors', type=str, default = 'custom_sfm_map.pkl')
    parser.add_argument('--colmap_map', type=str, default = 'colmap_truck_map.pkl')
    args = parser.parse_args()
    #Step 1: Get the descriptors out from the frame, and get the descriptors from the current frames as well
    K = np.array([[2688.0, 0, 1920.0], [0, 2688.0, 1080.0], [0, 0, 1.0]])
    map_loc = args.stored_descriptors
    with open(map_loc, "rb") as f:
        map_data = pickle.load(f)
    if os.path.exists(args.stored_descriptors):
        print("\n" + "="*30 + "\nCUSTOM MAP LOCALIZATION\n" + "="*30)
        with open(args.stored_descriptors, "rb") as f:
            custom_data = pickle.load(f)
        res_custom = custom_map_localization(args, custom_data)    
        report_metrics(res_custom, "Custom Map")
        plot_trajectories(custom_data, res_custom, title="Custom Map: Trajectory Comparison")
    
    if os.path.exists(args.colmap_map):
        print("\n" + "="*30 + "\nCOLMAP MAP LOCALIZATION\n" + "="*30)
        with open(args.colmap_map, "rb") as f:
            col_data = pickle.load(f)
        
        class MapPoint: pass
        reformatted_points = []
        for X, desc in zip(col_data["points_3d"], col_data["descriptors"]):
            mp = MapPoint()
            mp.X = X
            mp.descriptors = [desc]
            reformatted_points.append(mp)
        
        colmap_map_data = {"map_points": reformatted_points, "K": K, "camera_poses": col_data.get("camera_poses", [])}
        res_colmap = custom_map_localization(args, colmap_map_data)
        report_metrics(res_colmap, "COLMAP Map")
        plot_trajectories(colmap_map_data, res_colmap, title="COLMAP Map: Trajectory Comparison")   
     
     
     
        
if __name__ == "__main__":
    main()  
    
