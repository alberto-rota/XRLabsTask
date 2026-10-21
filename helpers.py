### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'helpers.py' collects reusable code or snippets that benefit code clarity when wrapped
"""

from turtle import left

import numpy as np
import cv2
import open3d as o3d  # Open3D for mesh operations and registrations


def anatomical_orientation(vox2pat: np.ndarray) -> str:
    # Positive-Negative axis have this labels, will be appended to the string based on the axis dirs found
    positive_labels = {0: "L", 1: "P", 2: "S"}
    negative_labels = {0: "R", 1: "A", 2: "I"}
    anatomical_orientation = ""

    for colidx, column in enumerate(vox2pat.T[:3]):
        # What row in each column has the max absoute value? (most of axis direction is aligned here)
        main_axis = np.argmax(np.abs(column[:3]))
        # Appending the label dependin on axis dir
        is_val_positive = column[main_axis] > 0
        anatomical_orientation += (
            positive_labels[main_axis]
            if is_val_positive
            else negative_labels[main_axis]
        )
    return anatomical_orientation


def find_transform_to_rpi(vox2pat: np.ndarray) -> np.ndarray:
    # create the transform that represents the RPI reference frame ()
    rpi_rf = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # Then I instantiate the PRI reference directions (vox2pat rotation part)
    pri_rf = np.eye(4)
    pri_rf[:3, :3] = vox2pat[:3, :3]

    # Computing the relative transform from PRI to RPI
    pri_to_rpi = np.linalg.inv(pri_rf) @ rpi_rf
    return pri_to_rpi


def compute_normals(
    mesh: o3d.geometry.TriangleMesh, ds: float = 3.0
) -> o3d.geometry.TriangleMesh:
    mesh.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=3 * ds, max_nn=30)
    )
    return mesh


def compute_fpfh_features(
    cloud: o3d.geometry.PointCloud, ds: float = 3.0
) -> o3d.pipelines.registration.Feature:
    """FPFH descriptors (local 3D shape signatures), one per point of an already downsampled cloud.
    The cloud MUST carry oriented normals:
    """
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=6 * ds, max_nn=100)
    )


def render_overlay(
    img,
    K_depth,
    T_mri2camera,
    overlay_scene,
    rect_size=(1920, 1080),
    color=(140, 172, 224),
    alpha=0.5,
    outline_color=None,
):
    """Shaded MRI mesh seen from the rectified left camera, blended onto img (BGR)."""
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K_depth),
        o3d.core.Tensor(T_mri2camera),
        rect_size[0],
        rect_size[1],
    )
    hit = overlay_scene.cast_rays(rays)
    mask = np.isfinite(hit["t_hit"].numpy())
    # Headlight (Lambert) shading: surfaces turned away from the camera get darker -> contours show up
    dirs = rays.numpy()[..., 3:]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    shade = np.clip(
        np.abs(np.sum(hit["primitive_normals"].numpy() * dirs, axis=-1)), 0.25, 1.0
    )

    overlay = img.copy()
    rendered = shade[mask, None] * np.array(color, dtype=np.float32)[None, :]
    overlay[mask] = (alpha * rendered + (1 - alpha) * img[mask]).astype(np.uint8)
    if outline_color is not None:
        # Silhouette: outer border of the rendered mesh
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(overlay, contours, -1, outline_color, 2, cv2.LINE_AA)
    return overlay


def fitness_gate(result, MIN_FITNESS=0.6, MAX_RMSE_mm=2.5):
    """Quality gate on an ICP result"""
    return result.fitness >= MIN_FITNESS and result.inlier_rmse <= MAX_RMSE_mm


def find_rigid(source_cloud, target_cloud, T_init, ds=3.0):
    """Coarse-to-fine point-to-plane ICP from T_init. Returns the last ICP result."""
    for max_dist in (2 * ds, ds):
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            max_dist,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=ds)
            ),
        )
        T_init = result.transformation
    return result

def depth_from_pair(left, right, map1_l, map2_l, map1_r, map2_r, sgbm, MIN_DISPARITY, NUM_DISPARITIES, FB):
    left_rect = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)

    # SGBM needs grayscale images. Converting
    gray = [
        cv2.cvtColor(
            cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for img in (left_rect, right_rect)
    ]
    # SGBM cannot match the first MIN_DISP+NUM_DISP columns. Padding
    pad = MIN_DISPARITY + NUM_DISPARITIES
    gray = [
        cv2.copyMakeBorder(g, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        for g in gray
    ]

    # Finding the disparity matches (SGBM call)
    disp = (
        sgbm.compute(*gray).astype(np.float32)[:, pad:] / 16.0
    )  # SGBM returns fixed-point disparity x16

    # Filtering: Some columns are ouyt of the right image width
    cols = np.arange(disp.shape[1], dtype=np.float32)[None, :]
    valid = (disp > MIN_DISPARITY) & (
        cols - disp >= 0
    )  # match must fall inside the real right image

    # Disparity to depth
    depth = np.zeros_like(disp)
    depth[valid] = FB / disp[valid]
    