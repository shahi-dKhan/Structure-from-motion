import cv2
import numpy as np
from utils.features import *
from utils.triangulation import *
from utils.helpers import *


class MapPoint:
    def __init__(self, X, descriptor, frame_ids=None, observations=None):
        """
        X           : 3D point (ndarray shape (3,))
        descriptor  : first 2D feature descriptor (ndarray)
        frame_ids   : list of frame indices this point is observed in
        observations: list of (frame_id, pt2d) tuples for reprojection tracking
        """
        self.X = X
        self.descriptors = [descriptor]
        self.observed_in = frame_ids if frame_ids is not None else []
        self.observations = observations if observations is not None else []


def initialize_map(X_init, desc1_inliers, desc2_inliers, R, t, pts1_inliers, pts2_inliers):
    """Build the initial two-frame map."""
    map_points = []
    camera_poses = []
    camera_poses.append((np.eye(3), np.zeros(3)))   # Frame 0
    camera_poses.append((R, t))                      # Frame 1
    for Xi, d1, d2, p1, p2 in zip(X_init, desc1_inliers, desc2_inliers, pts1_inliers, pts2_inliers):
        mp = MapPoint(
            Xi, d1,
            frame_ids=[0, 1],
            observations=[(0, p1), (1, p2)]
        )
        mp.descriptors.append(d2)
        map_points.append(mp)
    return map_points, camera_poses


def compute_global_reprojection_error(map_points, camera_poses, K):
    """Mean reprojection error across all observations in the map."""
    total_err = 0.0
    total_obs = 0
    for mp in map_points:
        X_h = np.append(mp.X, 1.0)
        for fid, pt2d in mp.observations:
            if fid >= len(camera_poses):
                continue
            R, t = camera_poses[fid]
            P = K @ np.hstack((R, t.reshape(3, 1)))
            proj = P @ X_h
            u, v = proj[0] / proj[2], proj[1] / proj[2]
            total_err += np.sqrt((u - pt2d[0]) ** 2 + (v - pt2d[1]) ** 2)
            total_obs += 1
    return total_err / total_obs if total_obs > 0 else 0.0


def estimate_pose_pnp(map_points, kp, desc, K):
    """
    Estimate camera pose via PnP against the current map.
    Uses averaged descriptors per map point, then refines with
    non-linear optimisation (useExtrinsicGuess=True).
    """
    if len(map_points) == 0:
        return None, None

    # Use the mean of all stored descriptors for each 3D point
    map_desc = np.array(
        [np.mean(mp.descriptors, axis=0) for mp in map_points],
        dtype=np.float32
    )

    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(map_desc, desc, k=2)

    pts3D, pts2D = [], []
    for match in raw_matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < 0.75 * n.distance:
            pts3D.append(map_points[m.queryIdx].X)
            pts2D.append(kp[m.trainIdx].pt)

    if len(pts3D) < 6:
        return None, None

    pts3D = np.array(pts3D, dtype=np.float32)
    pts2D = np.array(pts2D, dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3D, pts2D, K, None)
    if not success or inliers is None or len(inliers) < 6:
        return None, None

    # Refine pose on inliers using nonlinear LM optimisation
    inliers = inliers.flatten()
    _, rvec, tvec = cv2.solvePnP(
        pts3D[inliers],
        pts2D[inliers],
        K, None,
        rvec, tvec,
        useExtrinsicGuess=True
    )

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec.flatten()
    return R_new, t_new


def Expand_map(registered_frames, curr_frame_id, curr_kp, curr_desc,
               camera_poses, K, map_points, used_kps, reproj_threshold=2.0):
    """
    Triangulate new map points by matching the new camera against ALL
    previously registered cameras.

    registered_frames : list of (cam_idx, kp, desc) for every prior camera
    curr_frame_id     : camera_poses index of the current camera
    used_kps          : set of (cam_idx, kp_idx) already in the map;
                        updated in-place to prevent duplicate points.
    """
    bf = cv2.BFMatcher()
    R_curr, t_curr = camera_poses[curr_frame_id]
    P_curr = K @ np.hstack((R_curr, t_curr.reshape(3, 1)))

    # Pre-compute current camera centre for parallax check
    C_curr = -R_curr.T @ t_curr

    new_points = 0
    for prev_frame_id, prev_kp, prev_desc in registered_frames:
        raw_matches = bf.knnMatch(prev_desc, curr_desc, k=2)

        R_prev, t_prev = camera_poses[prev_frame_id]
        P_prev = K @ np.hstack((R_prev, t_prev.reshape(3, 1)))
        C_prev = -R_prev.T @ t_prev

        for match in raw_matches:
            if len(match) < 2:
                continue
            m, n = match
            if m.distance >= 0.75 * n.distance:
                continue

            # Skip features already incorporated into the map
            key_prev = (prev_frame_id, m.queryIdx)
            key_curr = (curr_frame_id, m.trainIdx)
            if key_prev in used_kps or key_curr in used_kps:
                continue

            pt_prev = prev_kp[m.queryIdx].pt
            pt_curr = curr_kp[m.trainIdx].pt

            X_lin = linear_triangulation(
                P_prev, P_curr,
                np.array([pt_prev]), np.array([pt_curr])
            )[0]
            X_refined = non_linear_refine_pt(
                np.append(X_lin, 1), P_prev, P_curr, pt_prev, pt_curr
            )

            # Parallax / baseline angle check (per triangulated point)
            ray1 = X_refined - C_prev
            ray2 = X_refined - C_curr
            norm1, norm2 = np.linalg.norm(ray1), np.linalg.norm(ray2)
            if norm1 < 1e-9 or norm2 < 1e-9:
                continue
            angle = np.arccos(np.clip(np.dot(ray1 / norm1, ray2 / norm2), -1.0, 1.0))
            if angle < np.deg2rad(1.0):
                continue
            if np.linalg.norm(X_refined) > 1000:
                continue
            # Cheirality check
            X_cam_prev = R_prev @ X_refined + t_prev
            X_cam_curr = R_curr @ X_refined + t_curr
            if X_cam_prev[2] <= 0 or X_cam_curr[2] <= 0:
                continue

            err_prev = reprojection_error(P_prev, np.array([X_refined]), np.array([pt_prev]))
            err_curr = reprojection_error(P_curr, np.array([X_refined]), np.array([pt_curr]))

            if err_prev < reproj_threshold and err_curr < reproj_threshold:
                mp = MapPoint(
                    X_refined,
                    curr_desc[m.trainIdx],
                    frame_ids=[prev_frame_id, curr_frame_id],
                    observations=[(prev_frame_id, pt_prev), (curr_frame_id, pt_curr)]
                )
                mp.descriptors.insert(0, prev_desc[m.queryIdx])
                map_points.append(mp)
                used_kps.add(key_prev)
                used_kps.add(key_curr)
                new_points += 1

    return new_points


def run_incremental_sfm(frame_stream, K, map_points, camera_poses,
                        registered_frames, max_frames=6):
    """
    registered_frames : list of (cam_idx, kp, desc) seeded with all cameras
                        registered so far (cam 0 and 1 from Task 2).
                        cam_idx is the index into camera_poses — NOT the video
                        frame number — so skipped frames never cause mismatches.
                        Grows as new frames are successfully added.
    """
    sift = cv2.SIFT_create()
    video_frame = len(camera_poses)   # video-frame counter (may skip values)
    used_kps = set()                  # (cam_idx, kp_idx) already in the map

    for img in frame_stream:
        if len(camera_poses) >= max_frames:
            break
        print(f"\nProcessing video frame {video_frame}...")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(img_gray, None)

        R_new, t_new = estimate_pose_pnp(map_points, kp, desc, K)
        if R_new is None:
            print(f"  Video frame {video_frame}: Pose estimation failed. Skipping.")
            video_frame += 1
            continue

        camera_poses.append((R_new, t_new))
        cam_idx = len(camera_poses) - 1  # stable index regardless of skips

        # Match against ALL previously registered cameras
        new_pts = Expand_map(
            registered_frames, cam_idx, kp, desc,
            camera_poses, K, map_points, used_kps
        )
        print(f"  New points added (vs {len(registered_frames)} prior cameras): {new_pts}")

        global_err = compute_global_reprojection_error(map_points, camera_poses, K)
        print(f"  Total cameras        : {len(camera_poses)}")
        print(f"  Total 3D points      : {len(map_points)}")
        print(f"  Mean reproj. error   : {global_err:.4f} px")

        # Register this camera so future frames can match against it
        registered_frames.append((cam_idx, kp, desc))
        video_frame += 1

    return map_points, camera_poses
