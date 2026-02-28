import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.features import get_camera_intrinsics, extract_and_match_features, estimate_essential_matrix, draw_epipolar_lines 
from utils.triangulation import decompose_essential_matrix, linear_triangulation, non_linear_refine_pt, disambiguate_poses

def reprojection_error(P, X, pts):
    total = 0
    for Xi, pi in zip(X, pts):
        Xi_h = np.append(Xi, 1)
        y = P @ Xi_h
        u = y[0] / y[2]
        v = y[1] / y[2]
        total += np.linalg.norm([u - pi[0], v - pi[1]])
    return total / len(X)



def extract_frames_from_video(video_path, interval=400):
    """
    Generator function that yields frames one by one in real-time.
    """
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
    video_file = '../Dataset/Split_A/split_a_truck-004.mp4'
    print(f"Streaming frames from {video_file} in real-time...")
    
    # Create the generator object
    frame_stream = extract_frames_from_video(video_file, interval=400)
    
    try:
        # 1. Grab the very first frame from the stream
        img1_color = next(frame_stream)
        K = get_camera_intrinsics(img1_color.shape)
        
        # 2. Grab the second frame from the stream to form our initial pair
        img2_color = next(frame_stream)
        
        print("Matching features...")
        img1_gray, img2_gray, kp1, kp2, desc1, desc2, pts1, pts2, good_matches = extract_and_match_features(img1_color, img2_color)
        
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
        print("Plotting epipolar lines...")
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
        err1 = reprojection_error(P1, X_linear, pts1_inliers)
        err2 = reprojection_error(P2, X_linear, pts2_inliers)
        print("Average reprojection error (linear):", (err1 + err2)/2)
        X_refined = []
        for Xi, p1, p2 in zip(X_linear, pts1_inliers, pts2_inliers):
            Xi_h = np.append(Xi, 1)
            Xr = non_linear_refine_pt(Xi_h, P1, P2, p1, p2)
            X_refined.append(Xr)
        X_refined = np.array(X_refined)
        err1_ref = reprojection_error(P1, X_refined, pts1_inliers)
        err2_ref = reprojection_error(P2, X_refined, pts2_inliers)
        print("Average reprojection error (refined):", (err1_ref + err2_ref)/2)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_linear[:,0], X_linear[:,1], X_linear[:,2], s=5)
        ax.set_title("Linear Triangulation")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_refined[:,0], X_refined[:,1], X_refined[:,2], s=5)
        ax.set_title("Refined Triangulation")
        plt.show()
        
        
    except StopIteration:
        print("Not enough frames extracted to perform matching.")