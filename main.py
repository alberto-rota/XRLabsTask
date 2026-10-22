### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'main.py' runs the whole pipeline:
    Part A: loads the DICOM volume, computes the transforms to the RPI convention and extracts the skin mesh ('mri.py')
    Part B: processes the stereo video into depth maps and point clouds ('surface_estimator.py'), registers them to
            the MRI skin mesh ('registration.py') and projects the registered mesh onto the left video frame.
    Visualization, metrics and disk outputs are handled by 'logger.py'

'python main.py --help' to see the args overrides
"""

import argparse
import os
import numpy as np
import cv2
from rich import print
from rich.progress import Progress  #
from time import time

# Importing the pipelines 
from mri import MRI # For DICOM loading, preprocessing and skin mesh extraction
from surface_estimator import SurfaceEstimator # For stereo rectification, matching and depth map generation
from registration import Registration # For registering the stereo point cloud to the MRI skin mesh
from logger import Logger # For logging metrics on Rerun and visualizations

print()

args = argparse.ArgumentParser()
args.add_argument(
    "--data_dir",
    type=str,
    default="data/Aneurysm_MRI",
    help="path to DICOM data directory, defaults to data/Aneurysm_MRI",
)

args.add_argument(
    "--video_path",
    type=str,
    default="data/surgical_navigation_task.avi",
    help="path to SBS stereo video",
)
args.add_argument(
    "--camera_params",
    type=str,
    default="data/camera_params.jsonc",
    help="path to the stereo camera parameters, defaults to data/camera_params.jsonc",
)
args.add_argument(
    "--part_a",
    action="store_true",
    help="run only Part A (DICOM, RPI conversion and skin mesh), defaults to False",
)
args.add_argument(
    "--body_volume_threshold",
    type=int,
    default=30,
    help="Intensity threshold for body volume in the skin mesh extraction process, defaults to 30",
)
args.add_argument(
    "--mri_cloud_size",
    type=int,
    default=100_000,
    help="size of the MRI cloud, defaults to 100_000",
)
args.add_argument(
    "--stride", type=int, default=1, help="process every N-th frame, defaults to 1"
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
    "--resolution_downsample",
    type=float,
    default=2,
    help="downsample factor K: stereo matching at 1/K resolution and point clouds at K mm voxels, defaults to 2 (8x faster than fullres)",
)
args.add_argument(
    "--viewer_image_res_down",
    type=int,
    default=3,
    help="downsample factor for the images sent to the Rerun viewer, defaults to 2 (4x faster than fullres)",
)
args.add_argument(
    "--depth_range_mm",
    type=int,
    nargs=2,
    default=(500, 1000),
    help="depth range in mm, defaults to 500 1000",
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
    "--save_overlays",
    action="store_true",
    help="save the overlay images to the disk as png, defaults to False",
)
args.add_argument(
    "--save_video_overlay",
    action="store_true",
    help="Export the overlay images as a full-resolution mp4 video, defaults to False",
)
args.add_argument(
    "--out_dir",
    type=str,
    default="out/",
    help="path outpiut directory, defaults to out/",
)

args = args.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

### PART A: MRI --------------------------------------------------------------------
mri = MRI(args.data_dir)
mri.save(args.out_dir)
mri.estimate_skin_surface(args.body_volume_threshold)
mri.save_skin_mesh(args.out_dir)

print(f"Opening Rerun viewer with 3D visualizations")
logger = Logger(
    out_dir=args.out_dir,
    save_frames=args.save_frames,
    save_depthmaps=args.save_depthmaps,
    save_overlays=args.save_overlays,
    focussed_tab=0 if args.part_a else 1,
)
logger.log_head_mesh(mri.vol3d_rpi, mri.verts, mri.faces, mri.normals)
if args.part_a:
    exit()

### PART B: LOADING THE VIDEO DATA ------------------------------------------------
print()
# Initializing the video reader
cap = cv2.VideoCapture(args.video_path)
flag, frame = cap.read()
if not flag:
    raise IOError(f"Could not read video {args.video_path}")
else:
    print(f"Video file opened: {args.video_path}")

# For downsampling = K, the voxel size becomes K mm (same on the video and MRI side, so the FPFH descriptors are comparable)
voxel = args.resolution_downsample
# Initializing the stereo surface estimator and the registration pipeline
surface_estimator = SurfaceEstimator(
    args.camera_params,
    frame_height=frame.shape[0],
    resolution_downsample=args.resolution_downsample,
    depth_range_mm=args.depth_range_mm,
    voxel=voxel,
)
logger.log_rectification_guides(surface_estimator.RECT_SIZE)
registration = Registration(
    mri.get_registration_target(args.mri_cloud_size, voxel),
    voxel=voxel,
)

# MAIN VIDEO STREAM LOOP -------------------------------------------------------------------------
# I set this up so that it could work completely online. The 'while True' below reflects it.

# Counting the total number of frames in the video. This is only for the sake of the progress bar.
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_frames is not None:
    total_frames = min(total_frames, args.max_frames)
frameidx = 0

# This wrapper is only so that we can have a progress bar in the terminal. I use it for debugging purposes.
with Progress() as progress:
    task = progress.add_task(logger.metrics_description(frameidx), total=total_frames)
    
    # While true - emulates a live stream
    while True:

        # Timing the whole frame processing, for logging purposes
        chrono_start = time()
        chrono_start_total = time()

        # Frame read
        ret, frame = cap.read()
        if not ret:
            break

        # First 50-ish frames are a static image. Can skip if the arg is set . We can also skip frames to speed up the processing, if the arg is set.
        if frameidx < args.skip_first_frames or frameidx % args.stride != 0:
            frameidx += 1
            continue
        
        # Extracting left and right frames
        left, right = surface_estimator.split_frame(frame)
        # Depth estimation pipeline: frame pair to depth
        depth, valid, left_rect, right_rect = surface_estimator.depth_from_pair(
            left, right
        )
        # Depthmap to point cloud in the camera RF.
        cloud_fromvideo = surface_estimator.get_registration_source(depth, valid)

        # Timers
        logger.update_mean(
            "avg_stereo_matching_time_ms", time() - chrono_start, frameidx
        )
        chrono_start = time()

        # Registration: Finds the transform that brings the point cloud estimated from this frame to the mri. Includes both the global ransa
        reg = registration.register(cloud_fromvideo)
        logger.update_mean(
            "avg_registration_time_ms", time() - chrono_start, frameidx
        )
        chrono_start = time()

        overlay = None
        if reg["T_camera2mri"] is not None:
            # The transform we found transforms the depthmap from the camera RF to the MRI RF.
            # The overlay transformation will need to bring the MRI to the camera RF, so we invert the transform.
            T_mri2camera = np.linalg.inv(reg["T_camera2mri"])

            # IMPORTANT: When the registration fails, THE OVERLAY SHOULD NOT BE DRAWN. STill, we draw it with a red contour to visualize the failure
            overlay = mri.render_from_pose(
                img=left_rect,
                K_depth=surface_estimator.K_depth,
                T_mri2camera=T_mri2camera,
                rect_size=surface_estimator.RECT_SIZE,
                outline_color=None if reg["registered"] else (0, 0, 255),
            )

        # Logging stuff on rerun. The frames, the depth, a 3D
        logger.log_frames(frameidx, overlay, left_rect, right_rect,frame_viz_downsample=args.viewer_image_res_down)
        logger.log_metrics(frameidx, reg)
        logger.log_depth_surface(
            depth,
            surface_estimator.K_depth,
            surface_estimator.RECT_SIZE,
            reg["T_camera2mri"],
        )
        logger.update_mean("avg_logging_time_ms", time() - chrono_start, frameidx)
        logger.update_mean(
            "avg_total_time_per_frame_ms", time() - chrono_start_total, frameidx
        )

        # Moving progress bar forward
        progress.update(
            task,
            completed=frameidx + 1,
            description=logger.metrics_description(frameidx),
        )

        # Saving frames, depthmaps and overlays to disk if requested
        logger.save_frame_outputs(frameidx, left_rect, right_rect, depth, overlay)

        frameidx += 1
        # In case we wanna process only a certain number of frames
        if args.max_frames is not None and frameidx >= args.max_frames:
            break

logger.summary()
cap.release()

# Saving output as full-res mp4
if args.save_video_overlay:
    print("> Exporting to full resolution mp4 video")
    logger.export_video_overlay()