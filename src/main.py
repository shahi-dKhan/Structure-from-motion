import numpy as np
import cv2
from utils.features import get_camera_intrinsics, extract_and_match_features, estimate_essential_matrix, draw_epipolar_lines 

def extract_frames_from_video(video_path, interval=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

if __name__ == "__main__":
    video_file = '../Dataset/Split_A/split_a_truck-004.mp4'
    print(f"Extracting frames from {video_file}...")
    
    frames = extract_frames_from_video(video_file, interval=10)
    print(f"Extracted {len(frames)} frames.")
    
    if len(frames) >= 2:
        img1_color = frames[0]
        img2_color = frames[1]
        K = get_camera_intrinsics(img1_color.shape)
        print("Matching features...")
        img1_gray, img2_gray, kp1, kp2, desc1, desc2, pts1, pts2, good_matches = extract_and_match_features(img1_color, img2_color)
        print("Estimating Essential Matrix...")
        E1, E2, mask = estimate_essential_matrix(pts1, pts2, K)
        
        print(f"Total Matches: {len(pts1)}")
        print(f"RANSAC Inliers: {np.sum(mask)}")
        
        K_inv = np.linalg.inv(K)
        F1 = K_inv.T @ E1 @ K_inv
        F2 = K_inv.T @ E2 @ K_inv
        
        print("Plotting epipolar lines...")
        draw_epipolar_lines(img1_color, img2_color, pts1, pts2, F1, title="E1: All Correspondences")
        draw_epipolar_lines(img1_color, img2_color, pts1, pts2, F2, title="E2: Robust (RANSAC)")
    else:
        print("Not enough frames extracted to perform matching.")
