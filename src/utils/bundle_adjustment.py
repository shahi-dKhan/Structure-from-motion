import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree

def build_observations(map_points):
    camera_indices = []
    point_indices = []
    points_2d = []
    filtered_points = []
    for mp in map_points:
        if len(mp.observations) < 2 :
            continue
        new_index = len(filtered_points)
        filtered_points.append(mp)
        for cam_idx, pt2d in mp.observations:
            camera_indices.append(cam_idx)
            points_2d.append(pt2d)
            point_indices.append(new_index)
    
    return (
        np.array(camera_indices),
        np.array(point_indices),
        np.array(points_2d, dtype=np.float64),
        filtered_points
        )
        
        
def pack_parameters(camera_poses, filtered_points):
    params = []
    for i in range(1, len(camera_poses)):
        R, t = camera_poses[i]
        rvec, _ = cv2.Rodrigues(R)
        params.extend(rvec.flatten())
        params.extend(t.flatten())
    
    for mp in filtered_points:
        params.extend(mp.X.flatten())
    return np.array(params)
    
def build_jacobian_sparsity(camera_indices, point_indices, n_cams, n_points):
    n_obs = len(camera_indices)
    n_cam_params = (n_cams - 1) * 6
    n_params = n_cam_params + n_points * 3
    
    A = lil_matrix((n_obs * 2, n_params), dtype=int)
    
    for i in range(n_obs):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        if cam_idx > 0:
            cam_offset = (cam_idx - 1) * 6
            A[2*i:2*i+2, cam_offset:cam_offset+6] = 1
        pt_offset = n_cam_params + pt_idx * 3
        A[2*i:2*i+2, pt_offset:pt_offset+3] = 1
        
    return A


def ba_residuals(params, n_cams, n_points, camera_indices, point_indices, points_2d, K):
    cam_param_count = 6 * (n_cams - 1)
    cam_params = params[:cam_param_count].reshape(-1, 6)
    points_3d = params[cam_param_count:].reshape(-1, 3)
    points_proj = np.zeros((len(points_2d), 2))
    for i in range(n_cams):
        mask = (camera_indices == i)
        if not np.any(mask):
            continue
        pts_3d_i = points_3d[point_indices[mask]]
        if i == 0:
            rvec = np.zeros(3,dtype=np.float64)
            t = np.zeros(3,dtype=np.float64)
        else:
            rvec = cam_params[i-1,:3]
            t = cam_params[i-1,3:]
        proj, _ = cv2.projectPoints(pts_3d_i, rvec, t, K, None)
        points_proj[mask] = proj.reshape(-1, 2)
    return (points_proj - points_2d).ravel()


    # for k in range(len(points_2d)):
    #     cam_idx = camera_indices[k]
    #     pt_idx = point_indices[k]
    #     u_obs, v_obs = points_2d[k]
    #     if cam_idx == 0:
    #         R = np.eye(3)
    #         t = np.zeros(3)
    #     else:
    #         cam_offset = (cam_idx - 1) * 6
    #         rvec = params[cam_offset:cam_offset + 3]
    #         t = params[cam_offset + 3:cam_offset + 6]
    #         R, _ = cv2.Rodrigues(rvec)
    #     pt_offset = cam_param_count + pt_idx*3
    #     X = params[pt_offset:pt_offset + 3]
        
    #     X_h = np.append(X, 1.0)
    #     P = K @ np.hstack((R,t.reshape(3,1)))
    #     proj = P @ X_h
    #     u_pred = proj[0]/proj[2]
    #     v_pred = proj[1]/proj[2]
    #     residuals.append(u_obs-u_pred)
    #     residuals.append(v_obs-v_pred)
    # return np.array(residuals)
    
    
def run_bundle_adjustment(map_points, camera_poses, K):
    print("\nRunning Bundle Adjustment...")
    camera_indices, point_indices, points_2d, filtered_points = build_observations(map_points)
    n_cams = len(camera_poses)
    n_points = len(map_points)
    print(f"  Cameras optimized : {n_cams - 1}")
    print(f"  Points optimized  : {n_points}")
    print(f"  Observations      : {len(points_2d)}")
    x0 = pack_parameters(camera_poses, filtered_points)
    print("  Building Jacobian sparsity matrix...")
    jac_sparsity = build_jacobian_sparsity(camera_indices, point_indices, n_cams, n_points)
    result = least_squares(
        ba_residuals, x0, 
        jac_sparsity=jac_sparsity, # MUST INCLUDE THIS
        verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
        args=(n_cams, n_points, camera_indices, point_indices, points_2d, K)
    )
    # result = least_squares(ba_residuals, x0, verbose=2, method='trf', args = (n_cams,n_points,camera_indices,point_indices,points_2d,K))
    print("Bundle Adjustment Finished.")
    optimized = result.x
    cam_param_count = 6 * (n_cams - 1)
    
    new_camera_poses = [camera_poses[0]]
    for i in range(1, n_cams):
        offset = (i - 1) * 6
        rvec = optimized[offset:offset + 3]
        t = optimized[offset + 3:offset + 6]
        R, _ = cv2.Rodrigues(rvec)
        new_camera_poses.append((R, t))
        
    for j in range(n_points):
        pt_offset = cam_param_count + j * 3
        filtered_points[j].X = optimized[pt_offset:pt_offset + 3]

    return filtered_points, new_camera_poses



def normalize_cloud(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    scale = np.mean(np.linalg.norm(centered_points, axis=1))
    normalized_points = centered_points / scale
    return normalized_points

def compute_chamfer_distance(cloud1, cloud2):
    tree1 = KDTree(cloud1)
    tree2 = KDTree(cloud2)
    dist1, _ = tree2.query(cloud1)
    dist2, _ = tree1.query(cloud2)
    return (np.mean(dist1**2) + np.mean(dist2**2)) / 2

