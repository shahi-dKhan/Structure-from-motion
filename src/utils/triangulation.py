import cv2
import numpy as np

def decompose_essential_matrix(E):
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=np.float64)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    return poses

def linear_triangulation(P1, P2, pts1, pts2):
    points_3d = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        points_3d.append(X[:3])
    return np.array(points_3d)


# def non_linear_refine_pt(X_init, P1, P2, pt1, pt2, iters=10):
#     if X_init.shape[0] == 4:
#         X = X_init[:3].copy()
#     else:
#         X = X_init.copy()

#     for _ in range(iters):
#         J = []
#         e = []
#         for P, pt in [(P1, pt1), (P2, pt2)]:
#             X_h = np.append(X, 1)
#             y = P @ X_h
#             u = y[0] / y[2]
#             v = y[1] / y[2]
#             e.extend([u - pt[0], v - pt[1]])
#             p1 = P[0]
#             p2 = P[1]
#             p3 = P[2]
#             du_dX = (p1[:3] * y[2] - y[0] * p3[:3]) / (y[2]**2)
#             dv_dX = (p2[:3] * y[2] - y[1] * p3[:3]) / (y[2]**2)
#             J.append(du_dX)
#             J.append(dv_dX)
#         J = np.array(J)
#         e = np.array(e)
#         delta_X, *_ = np.linalg.lstsq(J, -e, rcond=None)
#         X = X + delta_X
#     return X

def non_linear_refine_pt(X_init, Ps, pts, iters=10):
    """
    Refines a 3D point by minimizing reprojection error across N observing views.
    
    Ps  : list of 3x4 Projection matrices for each observing camera
    pts : list of (x,y) 2D pixel coordinates corresponding to the cameras
    """
    if X_init.shape[0] == 4:
        X = X_init[:3].copy()
    else:
        X = X_init.copy()

    for _ in range(iters):
        J = []
        e = []
        
        # Loop dynamically over all observing views
        for P, pt in zip(Ps, pts):
            X_h = np.append(X, 1)
            y = P @ X_h
            
            # Avoid division by zero if the point falls exactly on the camera center
            if abs(y[2]) < 1e-7: 
                continue
                
            u = y[0] / y[2]
            v = y[1] / y[2]
            
            e.extend([u - pt[0], v - pt[1]])
            
            p1 = P[0]
            p2 = P[1]
            p3 = P[2]
            
            du_dX = (p1[:3] * y[2] - y[0] * p3[:3]) / (y[2]**2)
            dv_dX = (p2[:3] * y[2] - y[1] * p3[:3]) / (y[2]**2)
            
            J.append(du_dX)
            J.append(dv_dX)
            
        if len(J) == 0:
            break
            
        J = np.array(J)
        e = np.array(e)
        
        delta_X, *_ = np.linalg.lstsq(J, -e, rcond=None)
        X = X + delta_X
        
    return X


def disambiguate_poses(poses, K, pts1, pts2):
    best_count = 0
    best_pose = None
    best_points = None
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))

    for R, t in poses:
        P2 = K @ np.hstack((R, t.reshape(3,1)))
        X = linear_triangulation(P1, P2, pts1, pts2)
        ZA = X[:,2]
        X_camB = (R @ X.T + t.reshape(3,1)).T
        ZB = X_camB[:,2]
        valid = np.sum((ZA > 0) & (ZB > 0))
        if valid > best_count:
            best_count = valid
            best_pose = (R, t)
            best_points = X
    return best_pose, best_points
        
