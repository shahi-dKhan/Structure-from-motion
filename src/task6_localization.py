#!/usr/bin/env python3
"""
Task 6: Camera Localization on Split B
======================================
Using the fixed 3D map from the custom pipeline (Tasks 1-4), localize
each sampled frame in Split B independently.

Constraints enforced:
  - Map is strictly fixed: no triangulation, map expansion, or BA.
  - Same feature extractor (SIFT) used in Tasks 1-4.
  - Localization failure threshold: < 6 PnP inliers.

Deliverables:
  (a) Reprojection error vs. frame index (temporal plot, mean ± std).
  (b) Inlier count / inlier ratio vs. frame index + correlation analysis.
  (c) 3D camera trajectory: Split A (mapping) in blue, Split B (localization) in red.

Usage:
  python task6_localization.py [--scenes truck barn meetingroom]
                               [--split_a_interval 120]
                               [--split_b_interval 30]
                               [--max_map_frames 8]
                               [--out_dir ../output/task6]
                               [--live_vis]
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; switch to "TkAgg" for pop-up windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3-D axes)
import numpy as np

# ---------------------------------------------------------------------------
# Allow running from either the repository root or src/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils.features import (
    get_camera_intrinsics,
    extract_and_match_features,
    estimate_essential_matrix,
)
from utils.triangulation import (
    decompose_essential_matrix,
    non_linear_refine_pt,
    disambiguate_poses,
)
from utils.helpers import reprojection_error
from utils.increment import (
    MapPoint,
    initialize_map,
    run_incremental_sfm,
    compute_global_reprojection_error,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_INLIERS = 6          # frames with fewer PnP inliers are marked as failures
LOWE_RATIO  = 0.75       # Lowe's ratio-test threshold (must match Tasks 1-4)

# FLANN KD-Tree parameters (SIFT descriptors are float32)
_FLANN_INDEX_KDTREE = 1
_FLANN_INDEX_PARAMS = dict(algorithm=_FLANN_INDEX_KDTREE, trees=5)
_FLANN_SEARCH_PARAMS = dict(checks=50)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
SCENES = [
    {
        "name":    "truck",
        "split_a": "../Dataset/Split_A/split_a_truck-004.mp4",
        "split_b": "../Dataset/Split_B/split_b_truck.mp4",
    },
    {
        "name":    "barn",
        "split_a": "../Dataset/Split_A/split_a_barn.mp4",
        "split_b": "../Dataset/Split_B/split_b_barn.mp4",
    },
    {
        "name":    "meetingroom",
        "split_a": "../Dataset/Split_A/split_a_meetingroom-005.mp4",
        "split_b": "../Dataset/Split_B/split_b_meetingroom.mp4",
    },
]


# ===========================================================================
# Helpers
# ===========================================================================

def _frame_generator(video_path: str, interval: int):
    """Yield (video_frame_number, frame_bgr) every *interval* video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            yield count, frame
        count += 1
    cap.release()


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera centre from (R, t)."""
    return -R.T @ t


# ===========================================================================
# Map building (Tasks 1-4 pipeline, replicated here for standalone use)
# ===========================================================================

def build_reference_map(
    split_a_path: str,
    interval: int = 120,
    max_frames: int = 8,
    verbose: bool = True,
) -> tuple:
    """
    Build the reference 3-D map from a Split A video.

    Parameters
    ----------
    split_a_path : str
        Path to the Split A .mp4 file.
    interval : int
        Sample one frame every *interval* video frames.
    max_frames : int
        Maximum number of cameras to register (including the seed pair).
    verbose : bool
        Print progress messages.

    Returns
    -------
    map_points : list[MapPoint]
    camera_poses : list[tuple[R, t]]
    K : np.ndarray  (3×3 camera intrinsic matrix)
    """
    gen = _frame_generator(split_a_path, interval)

    _, img1_color = next(gen)
    K = get_camera_intrinsics(img1_color.shape)
    _, img2_color = next(gen)

    if verbose:
        print("  [map] Extracting features and matching initial pair …")
    (_, _, kp1, kp2, desc1, desc2,
     pts1, pts2, _good,
     matched_desc1, matched_desc2) = extract_and_match_features(img1_color, img2_color)

    if verbose:
        print("  [map] Estimating Essential Matrix …")
    _E1, E2, mask = estimate_essential_matrix(pts1, pts2, K)
    mask = mask.astype(bool)

    pts1_in     = pts1[mask]
    pts2_in     = pts2[mask]
    desc1_in    = matched_desc1[mask]
    desc2_in    = matched_desc2[mask]

    if verbose:
        print(f"  [map] RANSAC inliers: {mask.sum()} / {len(pts1)}")

    poses = decompose_essential_matrix(E2)
    best_pose, X_linear = disambiguate_poses(poses, K, pts1_in, pts2_in)
    R_best, t_best = best_pose

    P1 = K @ np.hstack((np.eye(3),          np.zeros((3, 1))))
    P2 = K @ np.hstack((R_best, t_best.reshape(3, 1)))

    X_refined = []
    for Xi, p1, p2 in zip(X_linear, pts1_in, pts2_in):
        Xr = non_linear_refine_pt(np.append(Xi, 1), P1, P2, p1, p2)
        X_refined.append(Xr)
    X_refined = np.array(X_refined)

    map_points, camera_poses = initialize_map(
        X_refined, desc1_in, desc2_in, R_best, t_best, pts1_in, pts2_in
    )

    # Seed registered-frame list with both initial cameras (kp + desc of full frame)
    registered_frames = [(0, kp1, desc1), (1, kp2, desc2)]

    # Pass the remainder of the generator as the frame stream
    remaining_frames = (frame for _cnt, frame in gen)
    map_points, camera_poses = run_incremental_sfm(
        remaining_frames,
        K,
        map_points,
        camera_poses,
        registered_frames=registered_frames,
        max_frames=max_frames,
    )

    if verbose:
        err = compute_global_reprojection_error(map_points, camera_poses, K)
        print(f"  [map] Cameras: {len(camera_poses)}  |  "
              f"3-D points: {len(map_points)}  |  "
              f"mean reproj. error: {err:.4f} px")

    return map_points, camera_poses, K


# ===========================================================================
# Task 6 — Localization (strict: map is READ-ONLY)
# ===========================================================================

def _build_flann_index(map_points: list) -> tuple:
    """
    Build a FLANN KD-Tree index over the map descriptors.
    Called **once** per scene before the localization loop.

    Returns
    -------
    flann     : cv2.FlannBasedMatcher with the index already trained
    pts3D_map : (N, 3) float32 array of 3-D map point coordinates
    """
    map_desc = np.array(
        [np.mean(mp.descriptors, axis=0) for mp in map_points],
        dtype=np.float32,
    )
    pts3D_map = np.array([mp.X for mp in map_points], dtype=np.float32)

    flann = cv2.FlannBasedMatcher(_FLANN_INDEX_PARAMS, _FLANN_SEARCH_PARAMS)
    flann.add([map_desc])   # add map descriptors as the training set
    flann.train()           # build the KD-Tree index
    return flann, pts3D_map


def localize_frame(
    frame_bgr: np.ndarray,
    pts3D_map: np.ndarray,
    flann: cv2.FlannBasedMatcher,
    K: np.ndarray,
    sift: cv2.SIFT,
) -> tuple:
    """
    Localize a single frame against the *fixed* 3-D map using a
    pre-built FLANN KD-Tree index (approximate nearest-neighbour search).

    A single FLANN query covers both pose estimation and statistics:
      descriptor_new → KD-Tree → Lowe ratio filter → 2-D/3-D pairs
      → RANSAC-PnP  → LM refinement  → reprojection_error()

    Parameters
    ----------
    frame_bgr : BGR image
    pts3D_map : (N, 3) float32 — 3-D coordinates of map points (same order
                as the descriptors used to build *flann*)
    flann     : pre-built FlannBasedMatcher (trained on map descriptors)
    K         : camera intrinsics
    sift      : pre-created cv2.SIFT instance (reused for speed)

    Returns
    -------
    R          : np.ndarray(3,3) or None on failure
    t          : np.ndarray(3,)  or None on failure
    n_inliers  : int  (0 if PnP not attempted)
    n_total    : int  putative 2-D/3-D correspondences after Lowe filter
    reproj_err : float or None on failure
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)

    if desc is None or len(desc) == 0:
        return None, None, 0, 0, None

    # ------------------------------------------------------------------
    # 1. Approximate nearest-neighbour search via FLANN KD-Tree.
    #    query = new frame descriptors  |  train = pre-indexed map descs
    #    m.queryIdx → frame kp index   |  m.trainIdx → map point index
    # ------------------------------------------------------------------
    raw_matches = flann.knnMatch(desc, k=2)

    pts3D, pts2D = [], []
    for match in raw_matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < LOWE_RATIO * n.distance:
            pts3D.append(pts3D_map[m.trainIdx])
            pts2D.append(kp[m.queryIdx].pt)

    n_total = len(pts3D)
    if n_total < MIN_INLIERS:
        return None, None, n_total, n_total, None

    pts3D_np = np.array(pts3D, dtype=np.float32)
    pts2D_np = np.array(pts2D, dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. Robust pose estimation: RANSAC-PnP then LM nonlinear refinement.
    # ------------------------------------------------------------------
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3D_np, pts2D_np, K, None,
        iterationsCount=200,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < MIN_INLIERS:
        n_inl = 0 if inliers is None else len(inliers)
        return None, None, n_inl, n_total, None

    inliers   = inliers.flatten()
    n_inliers = len(inliers)

    # LM nonlinear refinement on inlier subset
    _, rvec, tvec = cv2.solvePnP(
        pts3D_np[inliers], pts2D_np[inliers], K, None,
        rvec, tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_loc, _ = cv2.Rodrigues(rvec)
    t_loc    = tvec.flatten()

    # ------------------------------------------------------------------
    # 3. Reprojection error — reuse reprojection_error() from helpers.py
    # ------------------------------------------------------------------
    P_loc = K @ np.hstack((R_loc, t_loc.reshape(3, 1)))
    err   = reprojection_error(P_loc, pts3D_np[inliers], pts2D_np[inliers])

    return R_loc, t_loc, n_inliers, n_total, err


def localize_split_b(
    video_path: str,
    map_points: list,
    K: np.ndarray,
    interval: int = 30,
    live_vis: bool = False,
) -> list:
    """
    Localize all sampled frames from a Split B video against the fixed map.

    Parameters
    ----------
    video_path : path to the Split B .mp4 file
    map_points : fixed 3-D map (never modified)
    K          : camera intrinsics
    interval   : sample one frame every *interval* video frames
    live_vis   : if True, display a live window with the localized pose overlay

    Returns
    -------
    results : list of (frame_idx, R, t, n_inliers, n_total, reproj_err)
              R is None for failed localizations.
    """
    sift    = cv2.SIFT_create()

    # Build the FLANN KD-Tree index once from all map descriptors.
    # Every frame query reuses this single pre-built index.
    print("  Building FLANN KD-Tree index over map descriptors …")
    flann, pts3D_map = _build_flann_index(map_points)
    print(f"  Index built over {len(pts3D_map)} map points.")

    results  = []
    failures = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_count  = 0   # raw video-frame counter
    sample_index = 0   # localized-frame counter (only sampled frames)

    if live_vis:
        cv2.namedWindow("Task 6 — Localization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Task 6 — Localization", 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_count % interval == 0:
            R, t, n_inl, n_tot, err = localize_frame(frame, pts3D_map, flann, K, sift)

            if R is None:
                failures += 1
                results.append((sample_index, None, None, n_inl, n_tot, None))
                status_str = f"FAIL (inliers={n_inl}/{n_tot})"
                color_bgr  = (0, 0, 220)
            else:
                results.append((sample_index, R, t, n_inl, n_tot, err))
                status_str = f"OK   inliers={n_inl}/{n_tot}  reproj={err:.2f}px"
                color_bgr  = (0, 200, 0)

            print(f"    Frame {sample_index:4d} (vid {video_count:6d}): {status_str}")

            # ---- Live visualization ----
            if live_vis:
                vis = frame.copy()
                cv2.putText(
                    vis, status_str, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2, cv2.LINE_AA,
                )
                if R is not None:
                    # Re-project a few map points as green dots
                    P_vis   = K @ np.hstack((R, t.reshape(3, 1)))
                    pts_vis = np.array([mp.X for mp in map_points[:200]])
                    for X3 in pts_vis:
                        X_h    = np.append(X3, 1.0)
                        proj   = P_vis @ X_h
                        if proj[2] <= 0:
                            continue
                        u, v   = int(proj[0] / proj[2]), int(proj[1] / proj[2])
                        h, w   = vis.shape[:2]
                        if 0 <= u < w and 0 <= v < h:
                            cv2.circle(vis, (u, v), 3, (0, 255, 0), -1)
                cv2.imshow("Task 6 — Localization", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            sample_index += 1

        video_count += 1

    cap.release()
    if live_vis:
        cv2.destroyAllWindows()

    total = len(results)
    print(
        f"  Localization: {total - failures}/{total} succeeded  "
        f"({failures} failures, failure-rate={failures / max(total, 1):.1%})"
    )
    return results


# ===========================================================================
# Deliverables
# ===========================================================================

def report_metrics(results: list, scene_name: str) -> dict:
    """
    Compute and print summary statistics from localization results.

    Returns a dict with keys: mean_err, std_err, mean_inliers,
    mean_inlier_ratio, n_success, n_fail, corr_inlier_err.
    """
    good = [(r[3], r[4], r[5]) for r in results if r[5] is not None]

    if not good:
        print(f"  [{scene_name}] No successful localizations — nothing to report.")
        return {}

    n_inl_arr  = np.array([g[0] for g in good])
    n_tot_arr  = np.array([g[1] for g in good])
    err_arr    = np.array([g[2] for g in good])

    inlier_ratio = n_inl_arr / np.maximum(n_tot_arr, 1)

    n_success = len(good)
    n_fail    = len(results) - n_success

    corr = np.corrcoef(n_inl_arr, err_arr)[0, 1] if len(n_inl_arr) > 1 else float("nan")

    print(f"\n  [{scene_name}] Localization Summary")
    print(f"    Frames localised : {n_success} / {len(results)}  (failures: {n_fail})")
    print(f"    Mean reproj err  : {np.mean(err_arr):.3f} px  ± {np.std(err_arr):.3f}")
    print(f"    Mean inlier count: {np.mean(n_inl_arr):.1f}")
    print(f"    Mean inlier ratio: {np.mean(inlier_ratio):.3f}")
    print(f"    Inlier–error corr: {corr:.3f}")

    return dict(
        mean_err         = float(np.mean(err_arr)),
        std_err          = float(np.std(err_arr)),
        mean_inliers     = float(np.mean(n_inl_arr)),
        mean_inlier_ratio= float(np.mean(inlier_ratio)),
        n_success        = n_success,
        n_fail           = n_fail,
        corr_inlier_err  = float(corr),
    )


def plot_reprojection_error(results: list, scene_name: str, out_dir: str = "."):
    """Temporal plot of reprojection error (Deliverable a)."""
    good = [(r[0], r[5]) for r in results if r[5] is not None]
    if not good:
        return

    frames, errs = zip(*good)
    errs = np.array(errs)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, errs, marker="o", markersize=3, linewidth=1, color="steelblue",
            label="reproj. error")
    ax.axhline(np.mean(errs), color="tomato", linewidth=1.5,
               linestyle="--", label=f"mean = {np.mean(errs):.2f} px")
    ax.fill_between(
        frames,
        np.mean(errs) - np.std(errs),
        np.mean(errs) + np.std(errs),
        alpha=0.15, color="tomato", label="±1 std",
    )
    ax.set_xlabel("Localized Frame Index")
    ax.set_ylabel("Reprojection Error (px)")
    ax.set_title(f"{scene_name} — Reprojection Error vs. Frame  "
                 f"(mean={np.mean(errs):.2f} ± {np.std(errs):.2f} px)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{scene_name}_reproj_error.png")
    fig.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_inlier_analysis(results: list, scene_name: str, out_dir: str = "."):
    """
    Inlier count, inlier ratio, and correlation with reprojection error
    (Deliverable b).
    """
    good = [(r[0], r[3], r[4], r[5]) for r in results if r[5] is not None]
    if not good:
        return

    frames, n_inl, n_tot, errs = zip(*good)
    n_inl   = np.array(n_inl)
    n_tot   = np.array(n_tot)
    errs    = np.array(errs)
    ratios  = n_inl / np.maximum(n_tot, 1)

    corr = np.corrcoef(n_inl, errs)[0, 1] if len(n_inl) > 1 else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(frames, n_inl, marker="o", markersize=3, linewidth=1,
                 color="steelblue")
    axes[0].set_ylabel("PnP Inlier Count")
    axes[0].set_title(f"{scene_name} — Inlier Analysis  (inlier–error corr: {corr:.3f})")
    axes[0].axhline(np.mean(n_inl), color="tomato", linestyle="--", linewidth=1,
                    label=f"mean = {np.mean(n_inl):.1f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(frames, ratios, marker="s", markersize=3, linewidth=1,
                 color="darkorange")
    axes[1].set_ylabel("Inlier Ratio")
    axes[1].axhline(np.mean(ratios), color="tomato", linestyle="--", linewidth=1,
                    label=f"mean = {np.mean(ratios):.3f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(frames, errs, marker="^", markersize=3, linewidth=1,
                 color="seagreen")
    axes[2].set_xlabel("Localized Frame Index")
    axes[2].set_ylabel("Reprojection Error (px)")
    axes[2].grid(True, alpha=0.35)

    # Mark frames with low inlier ratio (< 0.1) as weak-support regions
    weak_mask = ratios < 0.10
    for i, (f_idx, weak) in enumerate(zip(frames, weak_mask)):
        if weak:
            for ax in axes:
                ax.axvspan(f_idx - 0.5, f_idx + 0.5, color="red", alpha=0.10)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{scene_name}_inlier_analysis.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  Saved → {fpath}")


def _draw_frustum(
    ax,
    R: np.ndarray,
    t: np.ndarray,
    color: str,
    scale: float = 0.15,
    alpha: float = 0.7,
):
    """Draw a camera frustum on a 3-D axes."""
    C = _camera_center(R, t)

    # Four image-plane corners (relative to camera frame)
    corners_cam = np.array([
        [ 0.5,  0.5, 1.0],
        [-0.5,  0.5, 1.0],
        [-0.5, -0.5, 1.0],
        [ 0.5, -0.5, 1.0],
    ]) * scale

    # Rotate to world frame
    corners_world = [(R.T @ cc + C) for cc in corners_cam]

    # Pyramid edges: apex → four corners
    for corner in corners_world:
        ax.plot(
            [C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]],
            color=color, linewidth=0.7, alpha=alpha,
        )
    # Base rectangle
    for i in range(4):
        j = (i + 1) % 4
        c0, c1 = corners_world[i], corners_world[j]
        ax.plot(
            [c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]],
            color=color, linewidth=0.7, alpha=alpha,
        )


def plot_trajectory(
    map_points: list,
    camera_poses_a: list,
    results_b: list,
    scene_name: str,
    out_dir: str = ".",
):
    """
    3-D camera trajectory visualization (Deliverable c).
    Split A (mapping) cameras drawn in blue, Split B (localization) in red.
    Map points drawn in light grey.
    """
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection="3d")

    # --- Map points (background context) ---
    X_all   = np.array([mp.X for mp in map_points])
    norms   = np.linalg.norm(X_all, axis=1)
    keep    = norms < np.percentile(norms, 95)
    ax.scatter(
        X_all[keep, 0], X_all[keep, 1], X_all[keep, 2],
        s=1, c="lightgrey", alpha=0.4, label="Map points",
    )

    # --- Split A cameras ---
    centers_a = []
    for R, t in camera_poses_a:
        _draw_frustum(ax, R, t, color="royalblue", scale=0.15, alpha=0.85)
        centers_a.append(_camera_center(R, t))
    centers_a = np.array(centers_a)
    if len(centers_a) > 1:
        ax.plot(
            centers_a[:, 0], centers_a[:, 1], centers_a[:, 2],
            "b-", linewidth=1.5, label="Split A trajectory",
        )
    ax.scatter(
        centers_a[:, 0], centers_a[:, 1], centers_a[:, 2],
        c="blue", s=30, zorder=5,
    )

    # --- Split B cameras ---
    centers_b = []
    for (fidx, R, t, n_inl, n_tot, err) in results_b:
        if R is None:
            continue
        _draw_frustum(ax, R, t, color="tomato", scale=0.12, alpha=0.75)
        centers_b.append(_camera_center(R, t))

    if centers_b:
        centers_b = np.array(centers_b)
        if len(centers_b) > 1:
            ax.plot(
                centers_b[:, 0], centers_b[:, 1], centers_b[:, 2],
                "r-", linewidth=1.2, label="Split B trajectory",
            )
        ax.scatter(
            centers_b[:, 0], centers_b[:, 1], centers_b[:, 2],
            c="red", s=20, zorder=5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"{scene_name} — Camera Trajectories\n"
        f"Split A: {len(camera_poses_a)} cams (blue)  |  "
        f"Split B: {len(centers_b) if centers_b else 0} localised (red)"
    )
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{scene_name}_trajectory.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ===========================================================================
# Per-scene runner
# ===========================================================================

def run_scene(
    scene: dict,
    split_a_interval: int = 120,
    split_b_interval: int  = 30,
    max_map_frames: int    = 8,
    out_dir: str           = "../output/task6",
    live_vis: bool         = False,
) -> dict:
    """
    Full Task-6 pipeline for a single scene.
    Returns the metrics dict from report_metrics().
    """
    name = scene["name"]
    print(f"\n{'=' * 66}")
    print(f"  SCENE: {name.upper()}")
    print(f"{'=' * 66}")

    # ------------------------------------------------------------------
    # 1. Build the reference map from Split A
    # ------------------------------------------------------------------
    print(f"\n[1/3] Building reference map from Split A …")
    map_points, camera_poses_a, K = build_reference_map(
        scene["split_a"],
        interval   = split_a_interval,
        max_frames = max_map_frames,
        verbose    = True,
    )

    # ------------------------------------------------------------------
    # 2. Localize Split B frames (map is read-only from here on)
    # ------------------------------------------------------------------
    print(f"\n[2/3] Localizing Split B frames …")
    results = localize_split_b(
        scene["split_b"],
        map_points,
        K,
        interval = split_b_interval,
        live_vis = live_vis,
    )

    # ------------------------------------------------------------------
    # 3. Deliverables
    # ------------------------------------------------------------------
    print(f"\n[3/3] Computing metrics and generating plots …")
    metrics  = report_metrics(results, name)
    scene_out = os.path.join(out_dir, name)

    plot_reprojection_error(results, name, out_dir=scene_out)
    plot_inlier_analysis(results,    name, out_dir=scene_out)
    plot_trajectory(map_points, camera_poses_a, results, name, out_dir=scene_out)

    return metrics


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Task 6: Camera Localization on Split B"
    )
    parser.add_argument(
        "--scenes", nargs="+",
        choices=["truck", "barn", "meetingroom", "all"],
        default=["all"],
        help="Which scenes to process (default: all)",
    )
    parser.add_argument(
        "--split_a_interval", type=int, default=120,
        help="Sample every N-th frame from Split A for map building (default: 120)",
    )
    parser.add_argument(
        "--split_b_interval", type=int, default=30,
        help="Sample every N-th frame from Split B for localization (default: 30)",
    )
    parser.add_argument(
        "--max_map_frames", type=int, default=8,
        help="Max cameras to register during map building (default: 8)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="../output/task6",
        help="Output directory for plots (default: ../output/task6)",
    )
    parser.add_argument(
        "--live_vis", action="store_true",
        help="Show a live OpenCV window during localization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve scene list
    if "all" in args.scenes:
        target_scenes = SCENES
    else:
        target_scenes = [s for s in SCENES if s["name"] in args.scenes]

    all_metrics = {}
    for scene in target_scenes:
        all_metrics[scene["name"]] = run_scene(
            scene,
            split_a_interval = args.split_a_interval,
            split_b_interval = args.split_b_interval,
            max_map_frames   = args.max_map_frames,
            out_dir          = args.out_dir,
            live_vis         = args.live_vis,
        )

    # ------------------------------------------------------------------
    # Summary table across all scenes
    # ------------------------------------------------------------------
    print(f"\n{'=' * 66}")
    print("  SUMMARY")
    print(f"{'=' * 66}")
    print(f"  {'Scene':<14} {'Success':>8} {'Failures':>9} "
          f"{'Mean err (px)':>15} {'Mean inlier ratio':>18}")
    print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*15} {'-'*18}")
    for scene_name, m in all_metrics.items():
        if not m:
            print(f"  {scene_name:<14} {'N/A':>8}")
            continue
        print(
            f"  {scene_name:<14} "
            f"{m['n_success']:>8} "
            f"{m['n_fail']:>9} "
            f"{m['mean_err']:>15.3f} "
            f"{m['mean_inlier_ratio']:>18.3f}"
        )
    print(f"\nAll outputs saved to: {args.out_dir}")
