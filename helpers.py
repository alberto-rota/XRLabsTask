### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
    'helpers.py' collects reusable code or snippets that benefit code clarity when wrapped
"""

import numpy as np
import cv2

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

def compute_depth(left_rect, right_rect, sgbm, fb, depth_scale=1):
    # FB = -P2[0, 3]
    # Expected disparity at half resolution: ~185-205 px (depth ~520-580 mm). The search range [96, 288)
    # covers depths from ~370 mm to ~1100 mm.
    MIN_DISP, NUM_DISP = 96, 192
    """Depth map (mm) aligned to the half-resolution rectified left image, 0 where invalid."""
    gray = [
        cv2.cvtColor(cv2.resize(img, None, fx=depth_scale, fy=depth_scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        for img in (left_rect, right_rect)
    ]
    # SGBM cannot match the first MIN_DISP+NUM_DISP columns. Left-padding both images shrinks that blind zone
    # to the region the right camera truly does not see.
    pad = MIN_DISP + NUM_DISP
    gray = [cv2.copyMakeBorder(g, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0) for g in gray]
    disp = sgbm.compute(*gray).astype(np.float32)[:, pad:] / 16.0  # SGBM returns fixed-point disparity (x16)
    valid = disp > MIN_DISP
    depth = np.zeros_like(disp)
    depth[valid] = fb / (disp[valid] / depth_scale)  # disparity back to full rectified resolution
    return depth

def create_sgbm():
    """SGBM matcher with the parameters compute_depth assumes (disparity search [96, 288) at half resolution)."""
    MIN_DISP, NUM_DISP, BLOCK = 96, 192, 5
    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISP,
        numDisparities=NUM_DISP,  # must be divisible by 16
        blockSize=BLOCK,  # must be odd
        P1=24 * BLOCK**2,  # smoothness penalties, strong because the phantom surface is smooth
        P2=96 * BLOCK**2,
        disp12MaxDiff=1,  # left-right consistency check
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

def setup_stereo(cam: dict, depth_scale: float = 0.5) -> dict:
    """Rescaled intrinsics, rectification maps and depth-camera intrinsics for the SBS stereo video."""
    layout = cam["sbs_video_layout"]
    eye_width = layout["left_x_range"][1] - layout["left_x_range"][0]
    eye_size = (eye_width, layout["frame_height"])
    rect_size = (cam["calibration_resolution"]["width"], cam["calibration_resolution"]["height"])
    sx = eye_width / rect_size[0]

    def rescaled_K(K):
        K = np.array(K, dtype=float)
        K[0, 0] *= sx
        K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
        return K

    K1, d1 = rescaled_K(cam["left"]["K"]), np.array(cam["left"]["dist_opencv"])
    K2, d2 = rescaled_K(cam["right"]["K"]), np.array(cam["right"]["dist_opencv"])
    R, _ = cv2.Rodrigues(np.array(cam["stereo"]["rotation_vector"]))
    T = np.array(cam["stereo"]["translation_mm"]).reshape(3, 1)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, d1, K2, d2, eye_size, R, T, alpha=0, newImageSize=rect_size)
    # Intrinsics of the half-resolution rectified left camera, the one the depth maps are aligned to
    K_depth = P1[:3, :3].copy()
    K_depth[:2] *= depth_scale
    K_depth[:2, 2] += 0.5 * (depth_scale - 1)
    return {
        "left_x": slice(*layout["left_x_range"]),
        "right_x": slice(*layout["right_x_range"]),
        "maps_left": cv2.initUndistortRectifyMap(K1, d1, R1, P1, rect_size, cv2.CV_32FC1),
        "maps_right": cv2.initUndistortRectifyMap(K2, d2, R2, P2, rect_size, cv2.CV_32FC1),
        "P1": P1,
        "P2": P2,
        "fb": -P2[0, 3],
        "K_depth": K_depth,
        "depth_res": (int(rect_size[0] * depth_scale), int(rect_size[1] * depth_scale)),
    }
