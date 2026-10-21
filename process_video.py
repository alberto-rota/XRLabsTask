### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'process_video.py' processes the stereo video, rectifies the framepairs, computes the stereo matches, finds the disparity
map and ultimately the depth map in real world coordinates.

'python process_video.py --help' to see the args overrides

> Needs the outputs of 'dataload.py' and 'skin_segment.py'
> Provides outputs to ''
"""

import rerun as rr
import json5 as json
import argparse
import numpy as np
import os
import cv2  # OpenCV for loading avi video data
from rich import print
from rich.progress import Progress
import open3d as o3d
from helpers import *
from scipy.spatial import cKDTree
from time import time
print()

args = argparse.ArgumentParser()
args.add_argument(
    "--video",
    type=str,
    default="data/surgical_navigation_task.avi",
    help="path to SBS stereo video",
)
args.add_argument(
    "--mesh",
    type=str,
    default="out/skin_mesh.npz",
    help="path to skin mesh from skin_segment.py",
)
args.add_argument(
    "--mri_cloud_size",
    type=int,
    default=100_000,
    help="size of the MRI cloud, defaults to 150_000",
)
args.add_argument(
    "--stride", type=int, default=1, help="log every N-th frame to rerun, defaults to 1"
)
args.add_argument(
    "--max_frames",
    type=int,
    default=None,
    help="stop after N frames, defaults to the whole video",
)
args.add_argument(
    "--skip_first_frames",
    type=int,
    default=0,
    help="skip the first N frames, defaults to 0",
)
args.add_argument(
    "--match_res_downsample",
    type=float,
    default=0.5,
    help="downsample the resolution for the stereo matching, defaults to 0.5 (8x faster than fullres)",
)
args.add_argument(
    "--depth_range_mm",
    type=tuple,
    default=(500, 1000),
    help="depth range in mm, defaults to 500-1000mm",
)
args.add_argument(
    "--save_frames",
    action="store_true",
    help="save the frames to the disk, defaults to False",
)
args.add_argument(
    "--save_depthmaps",
    action="store_true",
    help="save the depth maps to the disk, defaults to False",
)
args.add_argument(
    "--out_dir",
    type=str,
    default="out/",
    help="path outpiut directory, defaults to out/",
)

args = args.parse_args()

### LOADING THE VIDEO DATA AND CAMERA PARAMETERS ------------------------------------------------
# Initializing the video reader and the camera parameters.
cap = cv2.VideoCapture(args.video)
flag, frame = cap.read()
if not flag:
    raise IOError(f"Could not read video {args.video}")
else:
    print(f"Video file opened: {args.video}")

# Loading the camera parameters as provided
cam = json.load(open("data/camera_params.jsonc"))
frame_height, frame_width = frame.shape[:2]
# Frame width gets overridden with the real measured value
frame_width = cam["sbs_video_layout"]["left_x_range"][1] - cam["sbs_video_layout"]["left_x_range"][0]

### ADJUSTING THE CAMERA INTRINSICS ------------------------------------------------
# Because the calibration images width was different to the vido frame wisth, the camera x-axis changes by a
# scaling factor (954/1920) and the principal point x-coordinate changes by a factor (652.2570/1920).
X_RESCALE_FACTOR = 954 / 1920

K1, d1 = np.array(cam["left"]["K"]), np.array(cam["left"]["dist_opencv"])
K1[0, 0] *= X_RESCALE_FACTOR
K1[0, 2] = X_RESCALE_FACTOR * (K1[0, 2] + 0.5) - 0.5

K2, d2 = np.array(cam["right"]["K"]), np.array(cam["right"]["dist_opencv"])
K2[0, 0] *= X_RESCALE_FACTOR
K2[0, 2] = X_RESCALE_FACTOR * (K2[0, 2] + 0.5) - 0.5

# Extrinsics
R, _ = cv2.Rodrigues(np.array(cam["stereo"]["rotation_vector"]))
T = np.array(cam["stereo"]["translation_mm"]).reshape(3, 1)

# Parameters to separate left and right eye images from the video stream
LEFT_X_END = slice(*cam["sbs_video_layout"]["left_x_range"])
RIGHT_X_START = slice(*cam["sbs_video_layout"]["right_x_range"])

### LOADING THE MRI DATA ---------------------------------------------------------------------
mesh_npz = np.load(args.mesh)

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(mesh_npz["verts"]),
    o3d.utility.Vector3iVector(mesh_npz["faces"]),
)

# Computing normals for the head mesh
mesh.compute_vertex_normals()
# MRI mesh to point cloud of the desired resolution (from the CLI)
mri_pcloud = mesh.sample_points_uniformly(
    number_of_points=args.mri_cloud_size, use_triangle_normal=True
)
mri_points = np.asarray(mri_pcloud.points)
mri_normals = np.asarray(mri_pcloud.normals)
print(f"MRI data loaded as mesh from '{args.mesh}")
print(
    f"{len(mri_points)} vertices & {len(mri_normals)} normals: {'[green]PASSED[/]' if len(mri_points) == len(mri_normals) else '[red]FAIL[/]'}"
)

# Make sure normals point outward (the normal direction should on  average be the same as the centerofmass-to-point direction)
if np.mean(np.sum((mri_points - mri_points.mean(axis=0)) * mri_normals, axis=1)) < 0:
    print("Head mesh normals are inward, flipping")
    mri_normals = -mri_normals
    mri_pcloud.normals = o3d.utility.Vector3dVector(mri_normals)

### STEREO RECTIFICATION MAPS ------------------------------------------------------------
# The video resolution is different than the calibration resolution (also, visually, the imagees are very squished horizontally).
# OpenCV stereorectify allows to specify the output resolution, we bribg it back to the calibration resolution

REAL_IMAGE_SIZE = (
    cam["calibration_resolution"]["width"],
    cam["calibration_resolution"]["height"],
)
RECT_SIZE = (
    int(REAL_IMAGE_SIZE[0] * args.match_res_downsample),
    int(REAL_IMAGE_SIZE[1] * args.match_res_downsample),
)
if args.match_res_downsample != 1:
    print(
        f"Running matching at {args.match_res_downsample} resolution. Expect {int(1 / args.match_res_downsample**3)}x speedup"
    )

# Finding the stereo rectification transforms
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, d1, K2, d2, (frame_width, frame_height), R, T, alpha=0, newImageSize=RECT_SIZE
)

# Depth = fB/disp --> fB we get from the rectified stereo output
FB = -P2[0, 3]
# Depth lives in the rectified left camera --> Its intrinsic matrix is P1
K_depth = P1[:3, :3]

# Pixel lookup tables for stereo rectufication (will be used by cv2.remap in the loop)
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, d1, R1, P1, RECT_SIZE, cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, d2, R2, P2, RECT_SIZE, cv2.CV_32FC1)
# Same for u_grid adn v_grid, which stays the same across the loop so we initialize it once here
v_grid, u_grid = np.mgrid[0 : RECT_SIZE[1], 0 : RECT_SIZE[0]]

### STEREO MATCHING---------------------------------------------------------------
MIN_DISPARITY = int(
    FB / args.depth_range_mm[1]
)  # Min disparity is the disparity for the maximum depth
NUM_DISPARITIES = int(
    FB / args.depth_range_mm[0] - MIN_DISPARITY
)  # Num disparities is the disparity range
BLOCK_SIZE = 7  # Very smooth surface --> strong smoothness penalties [Empirical]
SGBM_P1 = 24 * BLOCK_SIZE**2  # As suggested in the OpenCV documentation
SGBM_P2 = 96 * BLOCK_SIZE**2  # As suggested in the OpenCV documentation
# Intantiating the matcher. Will be called in the loop
sgbm = cv2.StereoSGBM_create(
    minDisparity=MIN_DISPARITY,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=SGBM_P1,
    P2=SGBM_P2,
    disp12MaxDiff=2,  # left-right consistency check
    speckleRange=3,  # Speckle filtering
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # 3-way matching to reduce noise
)
print("Running stereo matcher with parameters:")
print(f"  MIN_DISPARITY: {MIN_DISPARITY}")
print(f"  NUM_DISPARITIES: {NUM_DISPARITIES}")
print(f"  BLOCK_SIZE: {BLOCK_SIZE}")
print(f"  P1: {SGBM_P1}")
print(f"  P2: {SGBM_P2}")

### VISUALIZATION ---------------------------------------------------------------------
# Setting up the visualization first. This includes the rerun viewer and its blueprint and some
# static elements like the guides for the stereo matching.
rr.init("Stereo Rectification and Depth Map", spawn=True)
rr.send_blueprint(
    rr.blueprint.Horizontal(
        rr.blueprint.Vertical(
            # rr.blueprint.Horizontal(
            #     rr.blueprint.Spatial2DView(origin="/rect/left", name="Left (rectified)"),
            #     rr.blueprint.Spatial2DView(origin="/rect/right", name="Right (rectified)"),
            # ),
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial2DView(
                    origin="/overlay/left", name="Left (overlay)"
                ),
                # rr.blueprint.Spatial2DView(origin="/overlay/right", name="Right (overlay)"),
            ),
            rr.blueprint.TimeSeriesView(origin="/plots", name="Plots"),
        ),
        rr.blueprint.Vertical(
            rr.blueprint.Spatial3DView(
                origin="/mri",
                name="Depth point cloud",
            ),
            # rr.blueprint.Spatial3DView(origin="/depth", name="Depth point cloud",),
        ),
    )
)
# # Drawing horizontal lines at 60px intervals in the rectified images to visually check stereo rectification
# rows = np.arange(0, RECT_SIZE[1], 60)
# for view, width in (("/rect/left", RECT_SIZE[0]), ("/rect/right", RECT_SIZE[0])):
#     guides = rr.LineStrips2D([[[0, y], [width, y]] for y in rows], colors=[0, 255, 0], radii=0.5)
#     rr.log(f"{view}/guides", guides, static=True)
rr.log("/mri", rr.ViewCoordinates.LFD, static=True)
rr.log(
    "/mri/surface",
    rr.Mesh3D(
        vertex_positions=mesh_npz["verts"],
        triangle_indices=mesh_npz["faces"],
        vertex_normals=mesh_npz["normals"],
        albedo_factor=[224, 172, 140, 255],  # skin tone
    ),
    static=True,
)

# Preparing the scene to render the overlay of the MRI mesh that will be composed onto the video frame
overlay_scene = o3d.t.geometry.RaycastingScene()
overlay_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# REGISTRATION ----------------------------------------------------------------------
# Tracking strategy: every frame runs ICP starting from the previous pose (fast, precise, but local).
# The global FPFH+RANSAC registration (slow, robust, coarse) only runs when tracking is lost, followed by ICP.
# Every result (tracked or global) must pass a quality gate, otherwise the last good pose is kept.
MESH_DOWNSAMPLE = 2 
FOREGROUND_DEPTH_BAND_mm = 40  # Background removal: keep depths within this band from the nearest surface (the phantom)

# REGISTRATION CHECKS 
# Geometric constraint
def patch_motion(T_a, T_b, patch):
    """RMS distance (mm) the observed patch moves on the model between two camera->MRI poses."""
    D = T_a - T_b
    return np.sqrt(np.mean(np.sum((patch @ D[:3, :3].T + D[:3, 3]) ** 2, axis=1)))
MAX_PATCH_MOTION_mm = 6
MAX_LOST_FRAMES = 15  # the allowed motion grows with every lost frame, up to this many


def prepare_frame_cloud(points, voxel):
    """Video cloud at the given resolution, with normals oriented towards the camera"""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # Downsample before outlier removal: kNN on ~40k points instead of ~400k (~5x faster)
    cloud = cloud.voxel_down_sample(voxel)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30)
    )
    # Cloud is in camera coordinates, so the camera sits at the origin
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    return cloud

def ransac_near_pose(cloud, cloud_fpfh, T_prior, radius):
    """FPFH+RANSAC, with each video point only matched to MRI points within `radius` of where T_prior puts it.
    Keeps the re-initialization consistent with the motion prior (no flips on smooth, symmetric patches)."""
    predicted = np.asarray(cloud.points) @ T_prior[:3, :3].T + T_prior[:3, 3]
    feat = cloud_fpfh.data.T
    corr = [
        (i, idx[np.argmin(np.linalg.norm(mri_global_feat[idx] - feat[i], axis=1))])
        for i, idx in enumerate(mri_global_tree.query_ball_point(predicted, radius))
        if idx
    ]
    if len(corr) < 10:
        return T_prior
    return o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        cloud,
        mri_global_cloud,
        o3d.utility.Vector2iVector(np.array(corr, np.int32)),
        MESH_DOWNSAMPLE,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(MESH_DOWNSAMPLE),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
    ).transformation


# Same voxel size on both sides, so the descriptors are comparable
mri_global_cloud = mri_pcloud.voxel_down_sample(MESH_DOWNSAMPLE)
mri_global_cloud.normalize_normals()

mri_global_fpfh = compute_fpfh_features(mri_global_cloud, MESH_DOWNSAMPLE)
mri_global_xyz = np.asarray(mri_global_cloud.points)
mri_global_tree = cKDTree(mri_global_xyz)
mri_global_feat = mri_global_fpfh.data.T

print(
    f"Computed FPFH features from the MRI head mesh, with shape {mri_global_fpfh.data.shape}"
)

# MAIN LOOP -------------------------------------------------------------------------
# Counting the total number of frames in the video. This is only for the sake of the progress bar.
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_frames is not None:
    total_frames = min(total_frames, args.max_frames)
frameidx = 0
# Tracking state: last pose that passed the quality gate
T_refined = None
frames_since_good = 0  # grows while registration fails, widens the allowed motion

# Progress bar in the terminal to show how far along we are in the video processing
proc_metrics = {
    "avg_total_time_per_frame_ms": 0,
    "avg_stereo_matching_time_ms": 0,
    "avg_registration_time_ms": 0,
    "avg_logging_time_ms": 0,
    "total_registration_failures": 0,
}
def metrics_description(proc_metrics, frameidx):
    return (
        f"[cyan]M: {proc_metrics['avg_stereo_matching_time_ms']:.0f}ms, "
        f"R: {proc_metrics['avg_registration_time_ms']:.0f}ms, "
        f"L: {proc_metrics['avg_logging_time_ms']:.0f}ms, "
        f"T: {proc_metrics['avg_total_time_per_frame_ms']:.0f}ms, "
        f"F: {proc_metrics['total_registration_failures']}/{max(frameidx+1, 1)} ({proc_metrics['total_registration_failures'] / max(frameidx+1, 1):.0%})"
    )

def update_mean(key, seconds, frameidx):
    """Running mean in ms over the processed frames"""
    proc_metrics[key] += (seconds * 1000 - proc_metrics[key]) / (frameidx+1)
    
with Progress() as progress:
    task = progress.add_task(metrics_description(proc_metrics, frameidx), total=total_frames)
    while True:
        chrono_start = time()
        chrono_start_total = time()
        # Frame read
        ret, frame = cap.read()
        if not ret:
            break
        if frameidx < args.skip_first_frames:
            frameidx += 1
            continue

        if (
            frameidx % args.stride == 0
        ):  # In case we wanna process only every N-th frame
            # Extracting left and right frames and rectifying
            left, right = frame[:, LEFT_X_END], frame[:, RIGHT_X_START]

            depth = depth_from_pair(left, right=right, map1_l=map1_l, map2_l=map2_l, map1_r=map1_r, map2_r=map2_r, sgbm=sgbm, MIN_DISPARITY=MIN_DISPARITY, NUM_DISPARITIES=NUM_DISPARITIES, FB=FB)
            # Background removal: the phantom is the nearest surface, the table behind it is farther and noisier
            if valid.any():
                nearest_depth = np.percentile(
                    depth[valid], 2
                )  # Robust to a few too-near outliers
                valid &= depth < nearest_depth + FOREGROUND_DEPTH_BAND_mm
                depth[~valid] = 0

            # Depth to cloud
            points = np.stack(
                [
                    (u_grid[valid] - K_depth[0, 2]) * depth[valid] / K_depth[0, 0],
                    (v_grid[valid] - K_depth[1, 2]) * depth[valid] / K_depth[1, 1],
                    depth[valid],
                ],
                axis=1,
            )
            cloud_fromvideo = prepare_frame_cloud(points, MESH_DOWNSAMPLE)
            update_mean("avg_stereo_matching_time_ms", time() - chrono_start, frameidx)
            chrono_start = time()
            # Tracking: ICP from the previous pose, accepted only if the fit is good and the pose doesn't jump
            # Motion allowance: one frame of motion, plus one per lost frame (capped)
            frames_since_good += 1
            max_motion_mm = MAX_PATCH_MOTION_mm * min(frames_since_good, MAX_LOST_FRAMES)
            patch = np.asarray(cloud_fromvideo.points)[::10]

            def is_plausible(T_new):
                return T_refined is None or patch_motion(T_refined, T_new, patch) < max_motion_mm

            # Tracking: ICP from the previous pose, accepted only if the fit is good and the motion is plausible
            tracked = False
            if T_refined is not None:
                result = find_rigid(cloud_fromvideo, mri_global_cloud, T_refined)
                tracked = fitness_gate(result, MIN_FITNESS=0.45) and is_plausible(result.transformation)

            # Tracking lost: FPFH+RANSAC, then ICP. Searched near the last pose if there is one, globally otherwise
            registered = tracked
            if not tracked:
                cloud = prepare_frame_cloud(points, MESH_DOWNSAMPLE)
                frame_fpfh = compute_fpfh_features(cloud, MESH_DOWNSAMPLE)
                if T_refined is not None:
                    T_init = ransac_near_pose(cloud, frame_fpfh, T_refined, radius=max_motion_mm + 5)
                else:
                    ransac_fit_output = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source=cloud,
                        target=mri_global_cloud,
                        source_feature=frame_fpfh,
                        target_feature=mri_global_fpfh,
                        mutual_filter=True,
                        max_correspondence_distance=1 * MESH_DOWNSAMPLE,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                            False
                        ),
                        ransac_n=3,
                        checkers=[
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                                0.9
                            ),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                1 * MESH_DOWNSAMPLE
                            ),
                        ],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                            400_000, 0.999
                        ),
                    )
                    T_init = ransac_fit_output.transformation
                result = find_rigid(cloud_fromvideo, mri_global_cloud, T_init)
                registered = fitness_gate(result, MIN_FITNESS=0.75) and is_plausible(result.transformation)

            # Only a pose that passed the quality gate and the motion prior is kept. Otherwise the last good one is held
            if registered:
                T_refined = result.transformation
                frames_since_good = 0
            else:
                proc_metrics["total_registration_failures"] += 1

            if tracked:
                T_refined = result.transformation
            update_mean("avg_registration_time_ms", time() - chrono_start, frameidx)
            chrono_start = time()
            # The transform we found transforms the depthmap from the camera RF to the MRI RF.
            # The overlay transformation will need to bring the MRI to the camera RF, so we invert the transfor
            if T_refined is not None:
                T_mri2camera = np.linalg.inv(T_refined)
                # Red outline: this frame failed registration and shows the last good pose
                overlay = render_overlay(
                    img=left_rect,
                    K_depth=K_depth,
                    T_mri2camera=T_mri2camera,
                    overlay_scene=overlay_scene,
                    rect_size=RECT_SIZE,
                    color=(140, 172, 224),
                    alpha=0.5,
                    outline_color=None if registered else (0, 0, 255),
                )
                FRAME_DOWNSAMPLE = 2
                rr.log("/overlay/left", rr.Image(overlay[::FRAME_DOWNSAMPLE, ::FRAME_DOWNSAMPLE], color_model="BGR"))

            # Logging
            rr.set_time("frame", sequence=frameidx)
            rr.log("/plots/fitness", rr.Scalars(result.fitness))
            rr.log("/plots/inlier_rmse", rr.Scalars(result.inlier_rmse))
            rr.log("/plots/global_reinit", rr.Scalars(float(not tracked)))
            rr.log("/plots/registered", rr.Scalars(float(registered)))

            if T_refined is not None:
                rr.log(
                    "/mri/depth_registered",
                    rr.Transform3D(
                        translation=T_refined[:3, 3], mat3x3=T_refined[:3, :3]
                    ),
                )
            DEPTH_DOWNSAMPLE = 2
            K_depth_viz = K_depth.copy()
            K_depth_viz[0,0] = K_depth_viz[0,0] / DEPTH_DOWNSAMPLE
            K_depth_viz[1,1] = K_depth_viz[1,1] / DEPTH_DOWNSAMPLE
            K_depth_viz[0,2] = (K_depth_viz[0,2]-0.5) / DEPTH_DOWNSAMPLE +0.5
            K_depth_viz[1,2] = (K_depth_viz[1,2]-0.5) / DEPTH_DOWNSAMPLE +0.5
            depthviz=depth[::DEPTH_DOWNSAMPLE, ::DEPTH_DOWNSAMPLE]
            depthvizrange = depthviz[depthviz > 0].min(), depthviz.max()
            rr.log(
                "/mri/depth_registered/camera",
                rr.Pinhole(
                    image_from_camera=K_depth_viz,
                    resolution=RECT_SIZE,
                    image_plane_distance=20,
                ),
            )
            rr.log(
                "/mri/depth_registered/camera/image",
                rr.DepthImage(depth[::DEPTH_DOWNSAMPLE, ::DEPTH_DOWNSAMPLE], colormap="turbo",depth_range=depthvizrange)
            )

            # rr.log("/rect/left/image", rr.Image(left_rect, color_model="BGR"))
            # rr.log("/rect/right/image", rr.Image(right_rect, color_model="BGR"))
            # rr.log("/depth/cam/depth", rr.DepthImage(depth, colormap="turbo"))
            update_mean("avg_logging_time_ms", time() - chrono_start, frameidx)
            update_mean("avg_total_time_per_frame_ms", time() - chrono_start_total, frameidx)

            # Moving progress bar forward
            progress.update(task, completed=frameidx + 1,description=metrics_description(proc_metrics, frameidx))

            # Saving frames and depthmaps to npy files if requested
            if args.save_frames:
                os.makedirs(os.path.join(args.out_dir, "left"), exist_ok=True)
                np.save(
                    os.path.join(args.out_dir, "left", f"left_{frameidx:04d}.npy"),
                    left_rect,
                )
                os.makedirs(os.path.join(args.out_dir, "right"), exist_ok=True)
                np.save(
                    os.path.join(args.out_dir, "right", f"right_{frameidx:04d}.npy"),
                    right_rect,
                )
            if args.save_depthmaps:
                os.makedirs(os.path.join(args.out_dir, "depthmaps"), exist_ok=True)
                np.save(
                    os.path.join(
                        args.out_dir, "depthmaps", f"depth_{frameidx:04d}.npy"
                    ),
                    depth,
                )

            frameidx += 1
        # In case we wanna process only a certain number of frames
        if args.max_frames is not None and frameidx >= args.max_frames:
            break

# Checking filesaves
num_frames_saved = (
    len(os.listdir(os.path.join(args.out_dir, "frames"))) if args.save_frames else 0
)
num_depthmaps_saved = (
    len(os.listdir(os.path.join(args.out_dir, "depthmaps")))
    if args.save_depthmaps
    else 0
)

if args.save_frames:
    frame_save_ok = num_frames_saved == total_frames
    print(
        f"Saved {'[green]' if frame_save_ok else '[red]'} {num_frames_saved}/{total_frames} [/] frames to {os.path.join(args.out_dir, 'frames')}"
    )
if args.save_depthmaps:
    depthmap_save_ok = num_depthmaps_saved == total_frames
    print(
        f"Saved {'[green]' if depthmap_save_ok else '[red]'} {num_depthmaps_saved}/{total_frames} [/] depth maps to {os.path.join(args.out_dir, 'depthmaps')}"
    )
cap.release()
