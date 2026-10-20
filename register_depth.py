### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'register_depth.py' registers the stereo depth map of every video frame to the pre-operative skin mesh
and shows both in a shared 3D view (patient coordinates, mm).

Registration strategy:
1. Initialization on one keyframe: FPFH features + RANSAC (global, no initial guess needed), refined by
   point-to-plane ICP.
2. Tracking: every other frame is registered by ICP starting from the pose of the previous processed
   frame, forward and backward from the keyframe.
3. If tracking is lost (low ICP fitness), the global registration is attempted again on that frame.

'python register_depth.py --help' to see the args overrides

> Needs the outputs of 'dataload.py' and 'skin_segment.py'
> Provides outputs to ''
"""

import json
import argparse
import numpy as np
import cv2
import open3d as o3d  # Point cloud processing: features, global registration, ICP
import rerun as rr

from helpers import compute_depth, create_sgbm, setup_stereo

args = argparse.ArgumentParser()
args.add_argument(
    "--video",
    type=str,
    default="data/surgical_navigation_task.avi",
    help="path to SBS stereo video",
)
args.add_argument(
    "--camera_params",
    type=str,
    default="data/camera_params.json",
    help="path to camera parameters json",
)
args.add_argument(
    "--mesh",
    type=str,
    default="out/skin_mesh.npz",
    help="path to skin mesh from skin_segment.py",
)
args.add_argument(
    "--init_frame",
    type=int,
    default=1200,
    help="keyframe for the global registration, defaults to 1200 (nose, distinctive geometry)",
)
args.add_argument(
    "--stride", type=int, default=5, help="process every N-th frame, defaults to 5"
)
args.add_argument(
    "--max_frames",
    type=int,
    default=None,
    help="only process frames below this index, defaults to the whole video",
)
args = args.parse_args()

### PARAMETERS ---------------------------------------------------------------------------
VOXEL = 3.0  # mm, point cloud resolution used for features and ICP
DEPTH_BAND = 60.0  # mm, depth values further than this from the frame median are discarded as outliers
# Fitness = fraction of depth points within VOXEL mm of the mesh after ICP, RMSE = RMS distance of those points.
# A global registration (no prior) must be very convincing, smooth patches can fit the head in many wrong poses.
# Tracking is more lenient: background (table, instruments) entering the view lowers fitness even for a correct pose.
GLOBAL_MIN_FITNESS = 0.85
TRACK_MIN_FITNESS = 0.6
MAX_RMSE = 1.5  # mm
RESIDUAL_COLOR_MAX = 3.0  # mm, residual mapped to the top of the colormap
o3d.utility.random.seed(0)  # RANSAC reproducibility

### LOADING THE MESH ---------------------------------------------------------------------
mesh_npz = np.load(args.mesh)
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(mesh_npz["verts"]),
    o3d.utility.Vector3iVector(mesh_npz["faces"]),
)
mesh.compute_vertex_normals()
# Dense point sampling of the mesh as the registration target
target = mesh.sample_points_uniformly(number_of_points=150_000)
target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=3 * VOXEL, max_nn=30)
)
# Exact point-to-mesh distances for the residuals
raycast = o3d.t.geometry.RaycastingScene()
raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))


def fpfh(cloud):
    """Downsampled cloud with normals and its FPFH descriptors (local 3D shape signatures)."""
    down = cloud.voxel_down_sample(VOXEL)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=3 * VOXEL, max_nn=30)
    )
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=6 * VOXEL, max_nn=100)
    )
    return down, feat


target_down, target_feat = fpfh(target)

### STEREO SETUP -------------------------------------------------------------------------
cam = json.load(open(args.camera_params))
stereo = setup_stereo(cam)
sgbm = create_sgbm()
K_depth, depth_res = stereo["K_depth"], stereo["depth_res"]

# Overlays burned into the left video (zoom/focus HUD and crosshair) give wrong depth: mask them out.
# Boxes in left-eye pixels, warped to the rectified half-resolution depth image.
overlay = np.zeros(
    (
        cam["sbs_video_layout"]["frame_height"],
        stereo["left_x"].stop - stereo["left_x"].start,
    ),
    np.uint8,
)
overlay[0:140, 30:215] = 255  # HUD
overlay[500:580, 450:500] = 255  # crosshair
overlay = cv2.remap(overlay, *stereo["maps_left"], cv2.INTER_NEAREST)
overlay = (
    cv2.dilate(
        cv2.resize(overlay, depth_res, interpolation=cv2.INTER_NEAREST),
        np.ones((9, 9), np.uint8),
    )
    > 0
)

# Pixel grid of the depth image, for back-projection
v_grid, u_grid = np.mgrid[0 : depth_res[1], 0 : depth_res[0]]


def frame_to_cloud(frame):
    """Rectify, compute depth, clean it and back-project it to a camera-frame point cloud (mm)."""
    left_rect = cv2.remap(
        frame[:, stereo["left_x"]], *stereo["maps_left"], cv2.INTER_LINEAR
    )
    right_rect = cv2.remap(
        frame[:, stereo["right_x"]], *stereo["maps_right"], cv2.INTER_LINEAR
    )
    depth = compute_depth(left_rect, right_rect, sgbm, stereo["fb"])
    valid = (depth > 0) & ~overlay
    if valid.sum() > 0:
        valid &= np.abs(depth - np.median(depth[valid])) < DEPTH_BAND
    depth[~valid] = 0
    z = depth[valid]
    points = np.stack(
        [
            (u_grid[valid] - K_depth[0, 2]) * z / K_depth[0, 0],
            (v_grid[valid] - K_depth[1, 2]) * z / K_depth[1, 1],
            z,
        ],
        axis=1,
    )
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    left_small = cv2.resize(left_rect, depth_res, interpolation=cv2.INTER_AREA)
    return cloud, depth, left_small


### REGISTRATION -------------------------------------------------------------------------
def refine(source, T_init):
    """Coarse-to-fine point-to-plane ICP of the camera cloud onto the mesh samples."""
    T = T_init
    for max_dist in (4 * VOXEL, 2 * VOXEL, VOXEL):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_dist,
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        T = result.transformation
    return result


def global_register(cloud, trials=3):
    """FPFH + RANSAC initial alignment (no prior pose), each trial refined by ICP. Best ICP fitness wins."""
    source, source_feat = fpfh(cloud)
    best = None
    for _ in range(trials):
        ransac = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source,
                target_down,
                source_feat,
                target_feat,
                True,
                1.5 * VOXEL,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        1.5 * VOXEL
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(400_000, 0.999),
            )
        )
        result = refine(source, ransac.transformation)
        if best is None or result.fitness > best.fitness:
            best = result
    return best


def is_good(result, min_fitness):
    return result.fitness >= min_fitness and result.inlier_rmse <= MAX_RMSE


### PROCESSING ---------------------------------------------------------------------------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise IOError(f"Could not read video {args.video}")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_frames is not None:
    n_frames = min(n_frames, args.max_frames)
if not 0 <= args.init_frame < n_frames:
    raise ValueError(f"--init_frame must be within [0, {n_frames})")


def read_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


# Visualization setup: the world frame is the patient frame of the mesh (DICOM LPS, mm)
rr.init("depth_to_mesh_registration", spawn=True)
rr.send_blueprint(
    rr.blueprint.Horizontal(
        rr.blueprint.Spatial3DView(origin="/world", name="Mesh + registered depth"),
        rr.blueprint.Vertical(
            rr.blueprint.Spatial2DView(
                origin="/world/camera/image", name="Left (rectified)"
            ),
            rr.blueprint.Spatial2DView(origin="/depth", name="Depth (mm)"),
            rr.blueprint.TimeSeriesView(origin="/metrics", name="Registration quality"),
        ),
        column_shares=[2, 1],
    )
)
# LPS seen facing the patient: patient Left = viewer Right, Posterior = Forward (away), Superior = Up
rr.log("/world", rr.ViewCoordinates.RFU, static=True)
rr.log(
    "/world/skin",
    rr.Mesh3D(
        vertex_positions=mesh_npz["verts"],
        triangle_indices=mesh_npz["faces"].astype(np.uint32),
        vertex_normals=mesh_npz["normals"],
        albedo_factor=[200, 200, 200, 120],
    ),
    static=True,
)
rr.log(
    "/world/camera/image",
    rr.Pinhole(
        image_from_camera=K_depth, resolution=depth_res, image_plane_distance=40.0
    ),
    static=True,
)


def log_frame(idx, cloud, depth, left_small, result, status):
    rr.set_time("frame", sequence=idx)
    rr.log("/depth", rr.DepthImage(depth, meter=1.0, colormap="turbo"))
    rr.log(
        "/metrics/fitness", rr.Scalars(result.fitness if result is not None else np.nan)
    )
    rr.log(
        "/metrics/rmse_mm",
        rr.Scalars(result.inlier_rmse if result is not None else np.nan),
    )
    if status == "lost":
        # Clear the camera and points of the previous frame rather than showing a stale registration
        rr.log("/world/camera/image/rgb", rr.Clear(recursive=False))
        rr.log("/world/registered_depth", rr.Clear(recursive=False))
        return
    T = result.transformation  # camera -> patient
    rr.log("/world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    rr.log(
        "/world/camera/image/rgb",
        rr.Image(left_small, color_model="BGR").compress(jpeg_quality=85),
    )
    # Registered depth points, coloured by their distance to the mesh
    points = np.asarray(cloud.voxel_down_sample(1.0).points) @ T[:3, :3].T + T[:3, 3]
    residual = raycast.compute_distance(
        o3d.core.Tensor(points, dtype=o3d.core.float32)
    ).numpy()
    colors = cv2.applyColorMap(
        np.uint8(np.clip(residual / RESIDUAL_COLOR_MAX, 0, 1) * 255)[:, None],
        cv2.COLORMAP_TURBO,
    )[:, 0, ::-1]
    rr.log("/world/registered_depth", rr.Points3D(points, colors=colors, radii=0.4))
    rr.log("/metrics/median_residual_mm", rr.Scalars(np.median(residual)))


# 1) Global registration on the keyframe
frame = read_frame(args.init_frame)
cloud, depth, left_small = frame_to_cloud(frame)
init = global_register(cloud)
print(
    f"Keyframe {args.init_frame}: fitness {init.fitness:.2f}, RMSE {init.inlier_rmse:.2f} mm"
)
if not is_good(init, GLOBAL_MIN_FITNESS):
    raise RuntimeError(
        f"Global registration failed on keyframe {args.init_frame}, try another --init_frame"
    )
log_frame(args.init_frame, cloud, depth, left_small, init, "init")

# 2) Tracking forward and backward from the keyframe
forward = range(args.init_frame + args.stride, n_frames, args.stride)
backward = range(args.init_frame - args.stride, -1, -args.stride)
for frames in (forward, backward):
    T_prev = init.transformation
    for idx in frames:
        frame = read_frame(idx)
        if frame is None:
            break
        cloud, depth, left_small = frame_to_cloud(frame)
        result, status = None, "lost"
        if len(cloud.points) > 1000:
            source = cloud.voxel_down_sample(VOXEL)
            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=3 * VOXEL, max_nn=30)
            )
            if T_prev is not None:
                result = refine(source, T_prev)
                status = "tracked" if is_good(result, TRACK_MIN_FITNESS) else "lost"
            if status == "lost":
                # 3) Re-localization with the global registration, only accepted if convincing
                relocalized = global_register(cloud)
                if is_good(relocalized, GLOBAL_MIN_FITNESS):
                    result, status = relocalized, "relocalized"
                elif result is None:
                    result = relocalized
        T_prev = result.transformation if status != "lost" else None
        print(
            f"Frame {idx:5d}: {status:11s}"
            + (
                f" fitness {result.fitness:.2f}, RMSE {result.inlier_rmse:.2f} mm"
                if result
                else ""
            )
        )
        log_frame(idx, cloud, depth, left_small, result, status)
cap.release()
