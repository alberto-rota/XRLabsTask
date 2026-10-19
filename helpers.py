### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
    'helpers.py' collects reusable code or snippets that benefit code clarity when wrapped
"""

import numpy as np
import cv2
import open3d as o3d # Open3D for mesh operations and registrations

def anatomical_orientation(vox2pat: np.ndarray) -> str:
    # Positive-Negative axis have this labels, will be appended to the string based on the axis dirs found
    positive_labels = {0:'L',1:'P',2:'S'}
    negative_labels = {0:'R',1:'A',2:'I'}
    anatomical_orientation = ""

    for colidx, column in enumerate(vox2pat.T[:3]):
        # What row in each column has the max absoute value? (most of axis direction is aligned here)
        main_axis = np.argmax(np.abs(column[:3])) 
        # Appending the label dependin on axis dir
        is_val_positive = column[main_axis] > 0
        anatomical_orientation += positive_labels[main_axis] if is_val_positive else negative_labels[main_axis]
    return anatomical_orientation

def find_transform_to_rpi(vox2pat: np.ndarray) -> np.ndarray:
    # create the transform that represents the RPI reference frame ()
    rpi_rf = np.array([ [-1,  0,  0,  0],
                        [ 0,  1,  0,  0],
                        [ 0,  0, -1,  0],
                        [ 0,  0,  0,  1]])
    # Then I instantiate the PRI reference directions (vox2pat rotation part)
    pri_rf = np.eye(4)
    pri_rf[:3,:3] = vox2pat[:3,:3]

    # Computing the relative transform from PRI to RPI
    pri_to_rpi = np.linalg.inv(pri_rf) @ rpi_rf
    return pri_to_rpi

def compute_normals(mesh: o3d.geometry.TriangleMesh, ds: float = 3.0) -> o3d.geometry.TriangleMesh:
    mesh.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=3* ds, max_nn=30)
    )
    return mesh

def compute_fpfh_features(cloud: o3d.geometry.PointCloud, ds: float = 3.0) -> o3d.pipelines.registration.Feature:
    """FPFH descriptors (local 3D shape signatures), one per point of an already downsampled cloud.
    The cloud MUST carry oriented normals: 
    """
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=6 * ds, max_nn=100)
    )