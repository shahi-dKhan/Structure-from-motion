import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Feature caching helpers
# ---------------------------------------------------------------------------

def _keypoints_to_arrays(keypoints):
    """Serialize a list of cv2.KeyPoint objects into numpy arrays."""
    pts     = np.array([kp.pt       for kp in keypoints], dtype=np.float32)
    sizes   = np.array([kp.size     for kp in keypoints], dtype=np.float32)
    angles  = np.array([kp.angle    for kp in keypoints], dtype=np.float32)
    resps   = np.array([kp.response for kp in keypoints], dtype=np.float32)
    octaves = np.array([kp.octave   for kp in keypoints], dtype=np.int32)
    ids     = np.array([kp.class_id for kp in keypoints], dtype=np.int32)
    return pts, sizes, angles, resps, octaves, ids


def _arrays_to_keypoints(pts, sizes, angles, resps, octaves, ids):
    """Reconstruct cv2.KeyPoint objects from stored numpy arrays."""
    return [
        cv2.KeyPoint(x=float(pts[i, 0]), y=float(pts[i, 1]),
                     size=float(sizes[i]), angle=float(angles[i]),
                     response=float(resps[i]), octave=int(octaves[i]),
                     class_id=int(ids[i]))
        for i in range(len(pts))
    ]


def detect_and_compute_cached(img_colored, cache_path=None):
    """Detect SIFT keypoints and compute descriptors, with optional disk caching.

    If *cache_path* is provided and the corresponding ``.npz`` file already
    exists the features are loaded from disk instead of being recomputed.
    When the cache file does not yet exist the features are computed normally
    and then saved for future runs.

    Args:
        img_colored (np.ndarray): BGR image.
        cache_path (str | None): Path to the ``.npz`` cache file (the
            ``.npz`` extension is appended automatically if omitted).
            Pass ``None`` to skip caching entirely.

    Returns:
        tuple: (keypoints, descriptors) as returned by ``sift.detectAndCompute``.
    """
    if cache_path is not None:
        if not cache_path.endswith(".npz"):
            cache_path = cache_path + ".npz"
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            keypoints = _arrays_to_keypoints(
                data["kp_pts"], data["kp_sizes"], data["kp_angles"],
                data["kp_responses"], data["kp_octaves"], data["kp_class_ids"]
            )
            descriptors = data["descriptors"]
            return keypoints, descriptors

    img_gray = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        pts, sizes, angles, resps, octaves, ids = _keypoints_to_arrays(keypoints)
        np.savez_compressed(cache_path,
                            kp_pts=pts, kp_sizes=sizes, kp_angles=angles,
                            kp_responses=resps, kp_octaves=octaves,
                            kp_class_ids=ids, descriptors=descriptors)

    return keypoints, descriptors


def get_camera_intrinsics(image_shape):
    """_summary_

    Args:
        image_shape (np.ndarray): Contains the height and width of the image

    Returns:
        np.ndarray: The camera intrinsic matrix K
    """
    H, W = image_shape[:2]
    f = 0.7 * W
    cx, cy = W / 2, H / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype = np.float64)
    return K


def extract_and_match_features(img1_colored, img2_colored,
                               cache_path1=None, cache_path2=None):
    """_summary_

    Args:
        img1_colored (np.ndarray): the colored copy of image 1
        img2_colored (np.ndarray): the colored copy of image 2
        cache_path1 (str | None): optional path to the ``.npz`` cache file for
            image 1's SIFT features (e.g. ``"cache/video_frame_001"``).
        cache_path2 (str | None): optional path to the ``.npz`` cache file for
            image 2's SIFT features.

    Returns:
        tuple: (img1, img2, kp1, kp2, desc1, desc2, pts1, pts2, good_matches, matched_desc1, matched_desc2)
    """
    img1 = cv2.cvtColor(img1_colored, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_colored, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detect_and_compute_cached(img1_colored, cache_path1)
    kp2, desc2 = detect_and_compute_cached(img2_colored, cache_path2)
    bf2 = cv2.BFMatcher()
    matches = bf2.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    pts1, pts2 = [], []
    matched_desc1, matched_desc2 = [], []
    for match in matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            matched_desc1.append(desc1[m.queryIdx])
            matched_desc2.append(desc2[m.trainIdx])
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    matched_desc1 = np.array(matched_desc1)
    matched_desc2 = np.array(matched_desc2)
    return img1, img2, kp1, kp2, desc1, desc2, pts1, pts2, good_matches, matched_desc1, matched_desc2

def estimate_essential_matrix(pts1, pts2, K):
    """_summary_

    Args:
        pts1 (list): points from image 1 matched with points from image 2
        pts2 (list): points from image 2 with corresponding indices from image 1
        K (np.ndarray): The intrinsics matrix

    Raises:
        ValueError: If there are not enough points or the configuration is degenerate
        ValueError: If there are not enough points after the outliers are removed

    Returns:
        tuple: (E1, E2, mask) - The essential matrices and inlier mask
    """
    F1, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F1 is None:
        raise ValueError("findFundamentalMat returned None — not enough points or degenerate configuration.")
    E1_raw = K.T @ F1 @ K
    U, S, Vt = np.linalg.svd(E1_raw)
    S_rank2 = np.array([S[0], S[1], 0.0])
    E1 = U @ np.diag(S_rank2) @ Vt
    E2, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E2 is None or mask is None:
        raise ValueError("findEssentialMat returned None — not enough inliers or degenerate configuration.")
    return E1, E2, mask.ravel()

def draw_epipolar_lines(img1, img2, pts1, pts2, F, title = "Draw Epipolar Lines"):
    """_summary_

    Args:
        img1 (np.ndarray): The first image
        img2 (np.ndarray): The second image
        pts1 (list): Points from the first image
        pts2 (list): Points from the second image
        F (np.ndarray): The fundamental matrix
        title (str, optional): The title for the plot. Defaults to "Draw Epipolar Lines".
    """
    random_indices = np.random.choice(len(pts1), size=min(10, len(pts1)), replace=False)
    pts1_sample = pts1[random_indices]
    pts2_sample = pts2[random_indices]
    
    lines = cv2.computeCorrespondEpilines(pts1_sample.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1, 3)
    _, c, _ = img1.shape
    np.random.seed(42)
    img2_display = img2.copy()
    for r, pt1, pt2 in zip(lines, pts1_sample, pts2_sample):
        color = tuple(np.random.randint(0, 180, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img2_display = cv2.line(img2_display, (x0, y0), (x1, y1), color, 3)
        img2_display = cv2.circle(img2_display, tuple(map(int, pt2)), 15, color, -1)
        img2_display = cv2.circle(img2_display, tuple(map(int, pt2)), 15, (0, 0, 0), 1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img2_display, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
