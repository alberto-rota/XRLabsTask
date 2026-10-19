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
import json
import argparse
import numpy as np
import os
import cv2 # OpenCV for loading avi video data
from rich import print
from rich.progress import Progress
import open3d as o3d
from helpers import compute_fpfh_features
print()

args = argparse.ArgumentParser()
args.add_argument("--video", type=str, default="data/surgical_navigation_task.avi", help="path to SBS stereo video")
args.add_argument(
    "--mesh",
    type=str,
    default="out/skin_mesh.npz",
    help="path to skin mesh from skin_segment.py",
)
args.add_argument(
    "--mri_cloud_size",
    type=int,
    default=150_000,
    help="size of the MRI cloud, defaults to 150_000",
)
args.add_argument("--stride", type=int, default=1, help="log every N-th frame to rerun, defaults to 1")
args.add_argument("--max_frames", type=int, default=None, help="stop after N frames, defaults to the whole video")
args.add_argument("--match_res_downsample", type=float, default=0.5, help="downsample the resolution for the stereo matching, defaults to 0.5 (8x faster than fullres)")
args.add_argument("--depth_range_mm", type=tuple, default=(500, 1000), help="depth range in mm, defaults to 500-1000mm")
args.add_argument("--save_frames", action="store_true", help="save the frames to the disk, defaults to False")
args.add_argument("--save_depthmaps", action="store_true", help="save the depth maps to the disk, defaults to False")
args.add_argument("--out_dir", type=str, default="out/", help="path outpiut directory, defaults to out/")

args = args.parse_args()

### LOADING THE VIDEO DATA AND CAMERA PARAMETERS ------------------------------------------------
# Initializing the video reader and the camera parameters.  
cap = cv2.VideoCapture(args.video)
flag, frame = cap.read()
if not flag:
    raise IOError(f"Could not read video {args.video}")
else:
    print(f"Video file opened: {args.video}")
# Image frame dimensions
frame_height, frame_width = frame.shape[:2]
frame_width = 954 # IMPORTANT: Overridden with the real width value as specified in the notes

# Loading the camera parameters as provided
cam = json.load(open("data/camera_params.json"))

### ADJUSTING THE CAMERA INTRINSICS ------------------------------------------------
# Because the calibration images width was different to the vido frame wisth, the camera x-axis changes by a 
# scaling factor (954/1920) and the principal point x-coordinate changes by a factor (652.2570/1920).
X_RESCALE_FACTOR = 954/1920

K1, d1 = np.array(cam["left"]["K"]), np.array(cam["left"]["dist_opencv"])
K1[0,0] *= X_RESCALE_FACTOR
K1[0,2] = X_RESCALE_FACTOR * (K1[0,2] + 0.5) - 0.5 

K2, d2 = np.array(cam["right"]["K"]), np.array(cam["right"]["dist_opencv"])
K2[0,0] *= X_RESCALE_FACTOR
K2[0,2] = X_RESCALE_FACTOR * (K2[0,2] + 0.5) - 0.5 

# Extrinsics
R, _ = cv2.Rodrigues(np.array(cam["stereo"]["rotation_vector"]))
T = np.array(cam["stereo"]["translation_mm"]).reshape(3, 1) 

# Parameters to separate left and right eye images from the video stream
LEFT_X_END = slice(*cam["sbs_video_layout"]["left_x_range"])
RIGHT_X_START = slice(*cam["sbs_video_layout"]["right_x_range"])

### MRI DATALOADING ---------------------------------------------------------------------
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
# Make sure normals point outward
if np.mean(np.sum((mri_points - mri_points.mean(axis=0)) * mri_normals, axis=1)) < 0:
    mri_normals = -mri_normals
    mri_pcloud.normals = o3d.utility.Vector3dVector(mri_normals)
print(f"MRI data loaded as mesh from '{args.mesh}")
print(
    f"{len(mri_points)} vertices & {len(mri_normals)} normals: {'[green]PASSED[/]' if len(mri_points) == len(mri_normals) else '[red]FAIL[/]'}"
)

# Computing the static FPFH features from the MRI head mesh
# Same voxel size as the video frames, so descriptors are comparable. Averaged normals keep their (outward) sign
mri_pcloud = mri_pcloud.voxel_down_sample(3)
mri_pcloud.normalize_normals()
mri_fpfh = compute_fpfh_features(mri_pcloud)
print(f"Computed FPFH features from the MRI head mesh, with shape {mri_fpfh.data.shape}")

### STEREO RECTIFICATION MAPS ------------------------------------------------------------
# The video resolution is different than the calibration resolution (also, visually, the imagees are very squished horizontally).
# OpenCV stereorectify allows to specify the output resolution, we bribg it back to the calibration resolution

REAL_IMAGE_SIZE = (cam["calibration_resolution"]["width"], cam["calibration_resolution"]["height"])
RECT_SIZE = (int(REAL_IMAGE_SIZE[0] * args.match_res_downsample), int(REAL_IMAGE_SIZE[1] * args.match_res_downsample))
if args.match_res_downsample != 1:
    print(f"Running matching at {args.match_res_downsample} resolution. Expect {int(1/args.match_res_downsample**3)}x speedup")

# Finding the stereo rectification transforms
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, d1, K2, d2, (frame_width, frame_height), R, T, alpha=0, newImageSize=RECT_SIZE
)

# Depth = fB/disp --> fB we get from the rectified stereo output
FB = -P2[0,3]
# Depth lives in the rectified left camera --> P1
K_depth = P1[:3, :3]

# Pixel lookup tables for stereo rectufication
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, d1, R1, P1, RECT_SIZE, cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, d2, R2, P2, RECT_SIZE, cv2.CV_32FC1)

v_grid, u_grid = np.mgrid[0 : RECT_SIZE[1], 0 : RECT_SIZE[0]]

# At this point we only initialize the teansforms and the lookup tables. 
# They are applied to every frame in the loop below.
# We do the same for the stereo matcher 

### STEREO MATCHING---------------------------------------------------------------
# Instantiating the stereo matcher and its parameters.
DEPTH_RANGE_mm = (500, 1000) # ASSUMPTION: Depth range from 20 to 27cm (by visual inspection)
MIN_DISPARITY = int(FB / DEPTH_RANGE_mm[1]) # Min disparity is the disparity for the maximum depth
NUM_DISPARITIES = int(FB / DEPTH_RANGE_mm[0] - MIN_DISPARITY) # Num disparities is the disparity range
BLOCK_SIZE = 7 # Very smooth surface --> strong smoothness penalties
SGBM_P1 = 24 * BLOCK_SIZE**2 # As suggested in the OpenCV documentation
SGBM_P2 = 96 * BLOCK_SIZE**2 # As suggested in the OpenCV documentation

sgbm = cv2.StereoSGBM_create(
    minDisparity=MIN_DISPARITY,
    numDisparities=NUM_DISPARITIES, 
    blockSize=BLOCK_SIZE,
    P1=SGBM_P1,
    P2=SGBM_P2,
    disp12MaxDiff=1,  # left-right consistency check
    speckleRange=2, # Speckle filtering
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, # 3-way matching to reduce noise
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
    rr.blueprint.Vertical(
        # >> Uncomment if you want to visualize the raw images
        # rr.blueprint.Horizontal(
        #     rr.blueprint.Spatial2DView(origin="/raw/left", name="Left (raw)"),
        #     rr.blueprint.Spatial2DView(origin="/raw/right", name="Right (raw)"),
        # ),
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial2DView(origin="/rect/left", name="Left (rectified)"),
            rr.blueprint.Spatial2DView(origin="/rect/right", name="Right (rectified)"),
        ),
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial2DView(origin="/depth/cam", name="Depth (mm, rectified left)"),
            rr.blueprint.Spatial3DView(origin="/depth", name="Depth point cloud"),
        ),
    )
)
# Drawing horizontal lines at 60px intervals in the rectified images to visually check stereo rectification
rows = np.arange(0, RECT_SIZE[1], 60)
for view, width in (("/rect/left", RECT_SIZE[0]), ("/rect/right", RECT_SIZE[0])):
    guides = rr.LineStrips2D([[[0, y], [width, y]] for y in rows], colors=[0, 255, 0], radii=0.5)
    rr.log(f"{view}/guides", guides, static=True)
    
# Drawing the camera frustum. The depth map lives in the rectified left camera, so its intrinsics are P1 (not K1).
# The OpenCV convention (x right, y down, z forward) is rerun's default Pinhole convention (RDF).
rr.log("/depth/cam", rr.Pinhole(
    image_from_camera=P1[:3, :3],
    resolution=RECT_SIZE,
    camera_xyz=rr.ViewCoordinates.RDF,
    image_plane_distance=DEPTH_RANGE_mm[0]/2, # /2 is just for the sake of the visualization
), static=True)

# MAIN LOOP -------------------------------------------------------------------------

# Counting the total number of frames in the video. This is only for the sake of the progress bar.
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_frames is not None:
    total_frames = min(total_frames, args.max_frames)
frameidx = 0

# Progress bar in the terminal to show how far along we are in the video processing
with Progress() as progress:
    task = progress.add_task("[cyan]Processing video...", total=total_frames)
    frameidx = 0

    while True:
        # Frame read
        ret, frame = cap.read()
        if not ret:
            break

        if frameidx % args.stride == 0: # In case we wanna process only every N-th frame
            # Extracting left and right frames and rectifying
            left, right = frame[:, LEFT_X_END], frame[:, RIGHT_X_START]
            left_rect = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)
            
            # SGBM needs grayscale images. Converting
            gray = [
                cv2.cvtColor(cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                for img in (left_rect, right_rect)
            ]
            # SGBM cannot match the first MIN_DISP+NUM_DISP columns. Padding
            pad = MIN_DISPARITY + NUM_DISPARITIES
            gray = [cv2.copyMakeBorder(g, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0) for g in gray]

            # Finding the disparity matches (SGBM call)
            disp = sgbm.compute(*gray).astype(np.float32)[:, pad:] / 16.0 # SGBM returns fixed-point disparity x16

            # Filtering: Some columns are ouyt of the right image width
            cols = np.arange(disp.shape[1], dtype=np.float32)[None, :]
            valid = (disp > MIN_DISPARITY) & (cols - disp >= 0)  # match must fall inside the real right image

            # Disparity to depth
            depth = np.zeros_like(disp)
            depth[valid] = FB / disp[valid]  

            # Depth to cloud
            points = np.stack(
                [
                    (u_grid[valid] - K_depth[0, 2]) * depth[valid] / K_depth[0, 0],
                    (v_grid[valid] - K_depth[1, 2]) * depth[valid] / K_depth[1, 1],
                    depth[valid],
                ],
                axis=1,
            )
            cloud_fromvideo = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            FRAME_VOXEL_DOWNSAMPLE = 3
            # Downsample before outlier removal: kNN on ~40k points instead of ~400k (~5x faster)
            cloud_fromvideo = cloud_fromvideo.voxel_down_sample(FRAME_VOXEL_DOWNSAMPLE)
            cloud_fromvideo, _ = cloud_fromvideo.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            cloud_fromvideo.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=3 * FRAME_VOXEL_DOWNSAMPLE, max_nn=30)
            )
            # Cloud is in camera coordinates, so the camera sits at the origin
            cloud_fromvideo.orient_normals_towards_camera_location(np.zeros(3))

            frame_fpfh = compute_fpfh_features(cloud_fromvideo)
            ransac = (
                o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    cloud_fromvideo,
                    mri_pcloud,
                    frame_fpfh,
                    mri_fpfh,
                    True,
                    1.5 * FRAME_VOXEL_DOWNSAMPLE,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    3,
                    [
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                            0.9
                        ),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                            1.5 * FRAME_VOXEL_DOWNSAMPLE
                        ),
                    ],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(400_000, 0.999),
                )
            )
            T_ransac = ransac.transformation

            
            print(len(np.asarray(ransac.correspondence_set)),"/",len(frame_fpfh.data.T))
            rr.set_time("frame", sequence=frameidx)
            # >> Uncomment if you want to visualize the raw images
            # rr.log("/raw/left/image", rr.Image(left, color_model="BGR"))
            # rr.log("/raw/right/image", rr.Image(right, color_model="BGR"))
            rr.log("/rect/left/image", rr.Image(left_rect, color_model="BGR"))
            rr.log("/rect/right/image", rr.Image(right_rect, color_model="BGR"))
            rr.log("/depth/cam/depth", rr.DepthImage(depth, colormap="turbo"))
 
            # Moving progress bar forward
            progress.update(task, completed=frameidx + 1)

            # Saving frames and depthmaps to npy files if requested 
            if args.save_frames:
                os.makedirs(os.path.join(args.out_dir, "left"), exist_ok=True)
                np.save(os.path.join(args.out_dir, "left", f"left_{frameidx:04d}.npy"), left_rect)
                os.makedirs(os.path.join(args.out_dir, "right"), exist_ok=True)
                np.save(os.path.join(args.out_dir, "right", f"right_{frameidx:04d}.npy"), right_rect)
            if args.save_depthmaps:
                os.makedirs(os.path.join(args.out_dir, "depthmaps"), exist_ok=True)
                np.save(os.path.join(args.out_dir, "depthmaps", f"depth_{frameidx:04d}.npy"), depth)

        frameidx += 1
        # In case we wanna process only a certain number of frames
        if args.max_frames is not None and frameidx >= args.max_frames:
            break

# Checking filesaves
num_frames_saved = len(os.listdir(os.path.join(args.out_dir, "frames"))) if args.save_frames else 0
num_depthmaps_saved = len(os.listdir(os.path.join(args.out_dir, "depthmaps"))) if args.save_depthmaps else 0

if args.save_frames:
    frame_save_ok = num_frames_saved == total_frames
    print(f"Saved {'[green]' if frame_save_ok else '[red]'} {num_frames_saved}/{total_frames} [/] frames to {os.path.join(args.out_dir, 'frames')}")
if args.save_depthmaps:
    depthmap_save_ok = num_depthmaps_saved == total_frames
    print(f"Saved {'[green]' if depthmap_save_ok else '[red]'} {num_depthmaps_saved}/{total_frames} [/] depth maps to {os.path.join(args.out_dir, 'depthmaps')}")
cap.release()
