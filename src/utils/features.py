import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_camera_intrinsics(image_shape):
    H, W = image_shape[:2]
    f = 0.7 * W
    cx, cy = W / 2, H / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype = np.float64)
    return K


def extract_and_match_features(img1_colored, img2_colored):
    img1 = cv2.cvtColor(img1_colored, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_colored, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    
    # Feature Matching using Brute Force and Lowe's Ratio Test
    bf2 = cv2.BFMatcher()
    matches = bf2.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    return img1, img2, kp1, kp2, desc1, desc2, pts1, pts2, good_matches

def estimate_essential_matrix(pts1, pts2, K):
    F1, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    E1_raw = K.T @ F1 @ K
    U, S, Vt = np.linalg.svd(E1_raw)
    S_rank2 = np.array([S[0], S[1], 0.0])
    E1 = U @ np.diag(S_rank2) @ Vt
    E2, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E1, E2, mask.ravel()

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    pts1_sample = pts1[:10]
    pts2_sample = pts2[:10]
    
    lines = cv2.computeCorrespondEpilines(pts1_sample.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1, 3)
    _, c, _ = img1.shape
    np.random.seed(42)
    img2_display = img2.copy()
    for r, pt1, pt2 in zip(lines, pts1_sample, pts2_sample):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img2_display = cv2.line(img2_display, (x0, y0), (x1, y1), color, 1)
        img2_display = cv2.circle(img2_display, tuple(map(int, pt2)), 5, color, -1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img2_display, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines")
    plt.axis('off')
    plt.show()
     
