import numpy as np
import cv2

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
            yield count, frame  # yield (frame_number, frame)
            
        count += 1
        
    cap.release()
    
    
