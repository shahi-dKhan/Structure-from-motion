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

