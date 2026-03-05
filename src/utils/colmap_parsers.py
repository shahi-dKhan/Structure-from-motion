import numpy as np
import struct
import collections
import sqlite3
import cv2
import os

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_points3d_binary(path_to_model_file):
    """Reads COLMAP points3D.bin and returns a dictionary of 3D points."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_line_properties = struct.unpack("<QdddBBBd", fid.read(43))
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack(
                "<" + "ii" * track_length,
                fid.read(8 * track_length))
            image_ids = np.array(tuple(track_elems[0::2]))
            point2D_idxs = np.array(tuple(track_elems[1::2]))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D

def build_kd_tree_and_lookup(database_path, points3d_path):
    """
    Extracts descriptors from the database, links them to 3D points, 
    and builds a FLANN KD-Tree for Task 6.
    """
    print(f"Reading 3D points from {points3d_path}...")
    points3D = read_points3d_binary(points3d_path)
    
    print(f"Connecting to COLMAP database at {database_path}...")
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Extract all descriptors into a dictionary keyed by image_id
    cursor.execute("SELECT image_id, rows, cols, data FROM descriptors")
    image_descriptors = {}
    for image_id, rows, cols, data in cursor.fetchall():
        if data is not None:
            # COLMAP descriptors are stored as uint8
            desc = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
            image_descriptors[image_id] = desc
            
    conn.close()

    print("Building 2D-3D lookup table...")
    valid_3d_points = []
    descriptors_list = []
    
    # Link each 3D point to its corresponding descriptor
    for pt_id, pt in points3D.items():
        # A 3D point is observed by multiple cameras. We grab the first observation.
        img_id = pt.image_ids[0]
        pt2d_idx = pt.point2D_idxs[0]
        
        if img_id in image_descriptors:
            desc = image_descriptors[img_id][pt2d_idx]
            descriptors_list.append(desc)
            valid_3d_points.append(pt.xyz)
            
    descriptors_array = np.array(descriptors_list, dtype=np.float32)
    valid_3d_points = np.array(valid_3d_points)
    
    print(f"Extracted {len(valid_3d_points)} descriptors linked to 3D points.")
    
    # Build FLANN KD-Tree
    print("Building FLANN KD-Tree...")
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([descriptors_array])
    flann.train()
    print("KD-Tree built successfully!")
    
    return flann, valid_3d_points, descriptors_array