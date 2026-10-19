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
import cv2 # OpenCV for loading avi video data
from rich import print
from rich.progress import Progress

args = argparse.ArgumentParser()
args.add_argument("--video", type=str, default="data/surgical_navigation_task.avi", help="path to SBS stereo video")
args.add_argument("--stride", type=int, default=1, help="log every N-th frame to rerun, defaults to 1")
args.add_argument("--max_frames", type=int, default=None, help="stop after N frames, defaults to the whole video")
args.add_argument("--match_res_downsample", type=float, default=0.5, help="downsample the resolution for the stereo matching, defaults to 0.5 (8x faster than fullres)")
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

# Pixel lookup tables for stereo rectufication
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, d1, R1, P1, RECT_SIZE, cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, d2, R2, P2, RECT_SIZE, cv2.CV_32FC1)

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

            rr.set_time("frame", sequence=frameidx)
            # >> Uncomment if you want to visualize the raw images
            # rr.log("/raw/left/image", rr.Image(left, color_model="BGR"))
            # rr.log("/raw/right/image", rr.Image(right, color_model="BGR"))
            rr.log("/rect/left/image", rr.Image(left_rect, color_model="BGR"))
            rr.log("/rect/right/image", rr.Image(right_rect, color_model="BGR"))
            rr.log("/depth/cam/depth", rr.DepthImage(depth, colormap="turbo"))
 
            progress.update(task, completed=frameidx + 1)

        frameidx += 1
        # In case we wanna process only a certain number of frames
        if args.max_frames is not None and frameidx >= args.max_frames:
            break

cap.release()
