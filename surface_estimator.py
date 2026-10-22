### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
The SurfaceEstimator deals with everything that has to do with obtaining a point cloud from a stereo frame pair. It loads the camera parameters, finds the rectification transforms, creates the stereo matcher and
finds the disparity map and ultimately the depth map and point cloud in real world coordinates.

We use its constructor to load and initialize all the subroutines (rectification maps, stereo matching) so pay the computation upfront and then call the methods in the loop to get the point cloud from each frame.
"""

import json5 as json # Json5 because I added a comment in the json
import numpy as np
import cv2  # OpenCV for rectification and stereo matching
from rich import print
import open3d as o3d # Open3D for mesh and point cloud operations


class SurfaceEstimator:
    def __init__(
        self,
        camera_params,
        frame_height,
        resolution_downsample=2,
        depth_range_mm=(500, 1000),
        voxel=2,
        foreground_depth_band_mm=40,
    ):
        """Loads the stereo calibration, builds the rectification maps and instantiates the SGBM matcher

        camera_params: path to the calibration json (intrinsics, distortion, extrinsics, side-by-side video layout)
        frame_height: height in pixels of the video frames
        resolution_downsample: factor K, matching runs at 1/K of the calibration resolution (larger = faster matching)
        depth_range_mm: (min, max) working depth, sets the SGBM disparity search range
        voxel: point cloud resolution in mm, sets the normal estimation radius
        foreground_depth_band_mm: depths further than this from the nearest surface are treated as background
        """

        ### CAMERA PARAMETERS ------------------------------------------------------------
        cam = json.load(open(camera_params))
        # Frame width gets overridden with the real measured value
        frame_width = cam["sbs_video_layout"]["left_x_range"][1] - cam["sbs_video_layout"]["left_x_range"][0]
 
        # Because the calibration images width was different to the vido frame wiDth, the camera x-axis changes by a
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
        self.LEFT_X_END = slice(*cam["sbs_video_layout"]["left_x_range"])
        self.RIGHT_X_START = slice(*cam["sbs_video_layout"]["right_x_range"])

        ### STEREO RECTIFICATION MAPS ------------------------------------------------------------
        # The video resolution is different than the calibration resolution (also, visually, the imagees are very squished horizontally).
        # OpenCV stereorectify allows to specify the output resolution, we bribg it back to the calibration resolution
        REAL_IMAGE_SIZE = (
            cam["calibration_resolution"]["width"],
            cam["calibration_resolution"]["height"],
        )
        self.RECT_SIZE = (
            int(REAL_IMAGE_SIZE[0] / resolution_downsample),
            int(REAL_IMAGE_SIZE[1] / resolution_downsample),
        )
        if resolution_downsample != 1:
            print(
                f"Running matching at 1/{resolution_downsample} resolution. Expect {int(resolution_downsample**3)}x speedup"
            )

        # Finding the stereo rectification transforms
        R1, R2, P1, P2, _,_,_ = cv2.stereoRectify(
            K1, d1, K2, d2, (frame_width, frame_height), R, T, alpha=0, newImageSize=self.RECT_SIZE
        )

        # Depth = fB/disp --> fB we get from the rectified stereo output
        self.FB = -P2[0, 3]
        # Depth lives in the rectified left camera --> Its intrinsic matrix is P1. K_depth is used for depth-to-cloud conversion and visualization puposes
        self.K_depth = P1[:3, :3]
        print("Rectified left camera intrinsics (depth camera):")
        print(self.K_depth)
        print(f"Rectified baseline: {self.FB / self.K_depth[0, 0]:.2f}mm")

        # Pixel lookup tables for stereo rectufication (will be used by cv2.remap in the video stream loop)
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(K1, d1, R1, P1, self.RECT_SIZE, cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(K2, d2, R2, P2, self.RECT_SIZE, cv2.CV_32FC1)
        # Same for u_grid adn v_grid, which stays the same across the loop so we initialize it once here
        self.v_grid, self.u_grid = np.mgrid[0 : self.RECT_SIZE[1], 0 : self.RECT_SIZE[0]]

        ### STEREO MATCHER---------------------------------------------------------------
        self.MIN_DISPARITY = int(self.FB / depth_range_mm[1])  # Min disparity is the disparity for the maximum depth
        self.NUM_DISPARITIES = int(self.FB / depth_range_mm[0] - self.MIN_DISPARITY)  # Num disparities is the disparity range
        BLOCK_SIZE = 7  # Very smooth surface --> strong smoothness penalties [Empirical]
        SGBM_P1 = 24 * BLOCK_SIZE**2  # As suggested in the OpenCV documentation
        SGBM_P2 = 96 * BLOCK_SIZE**2  # As suggested in the OpenCV documentation
        # Intantiating the matcher. Will be called in the loop
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=self.MIN_DISPARITY,
            numDisparities=self.NUM_DISPARITIES,
            blockSize=BLOCK_SIZE,
            P1=SGBM_P1,
            P2=SGBM_P2,
            disp12MaxDiff=2,  # left-right consistency check
            speckleRange=3,  # Speckle filtering
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # 3-way matching to reduce noise
        )
        print("Running stereo matcher with parameters:")
        print(f"  MIN_DISPARITY: {self.MIN_DISPARITY}")
        print(f"  NUM_DISPARITIES: {self.NUM_DISPARITIES}")
        print(f"  BLOCK_SIZE: {BLOCK_SIZE}")
        print(f"  P1: {SGBM_P1}")
        print(f"  P2: {SGBM_P2}")

        self.voxel = voxel
        self.foreground_depth_band_mm = foreground_depth_band_mm  # Background removal: keep depths within this band from the nearest surface (the phantom)

    def split_frame(self, frame):
        """Splits a side-by-side stereo video frame into its (left, right) eye images"""
        return frame[:, self.LEFT_X_END], frame[:, self.RIGHT_X_START]

    def depth_from_pair(self, left, right):
        """Rectifies a stereo pair, runs SGBM and converts the disparity to depth (mm) via depth = fB/disparity

        Returns (depth, valid, left_rect, right_rect): the depth map in the rectified left camera, the mask of pixels
        with a valid match, and the two rectified BGR images
        """
        left_rect = cv2.remap(left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        # SGBM needs grayscale images. Converting
        gray = [
            cv2.cvtColor(
                cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
            for img in (left_rect, right_rect)
        ]
        # SGBM cannot match the first MIN_DISP+NUM_DISP columns. Padding
        pad = self.MIN_DISPARITY + self.NUM_DISPARITIES
        gray = [
            cv2.copyMakeBorder(g, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
            for g in gray
        ]

        # Finding the disparity matches (SGBM call)
        disp = (
            self.sgbm.compute(*gray).astype(np.float32)[:, pad:] / 16.0
        )  # SGBM returns fixed-point disparity x16

        # Filtering: Some columns are ouyt of the right image width
        cols = np.arange(disp.shape[1], dtype=np.float32)[None, :]
        valid = (disp > self.MIN_DISPARITY) & (
            cols - disp >= 0
        )  # match must fall inside the real right image

        # Disparity to depth via baseline
        depth = np.zeros_like(disp)
        depth[valid] = self.FB / disp[valid]
        return depth, valid, left_rect, right_rect

    def get_registration_source(self, depth, valid):
        """Turns a depth map into the point cloud that gets registered to the MRI

        Keeps only the foreground (within foreground_depth_band_mm of the nearest surface), back-projects the valid
        pixels with K_depth, removes statistical outliers and estimates normals oriented towards the camera.
        The cloud is in rectified left camera coordinates (mm). Note: valid and depth are modified in place.
        """

        # IMPORTANT LIMITATION: Video also contains background pixels which will be in the depthmap and 1. introduce noise 2. Wont have any target surface to match on the MRI. WE simply remove all pixels that are not in the foreground, which we empirically set to 40mm 
        if valid.any():
            nearest_depth = np.percentile(
                depth[valid], 2
            )  # Robust to a few too-near outliers
            valid &= depth < nearest_depth + self.foreground_depth_band_mm
            depth[~valid] = 0

        # Depth to cloud
        points = np.stack(
            [
                (self.u_grid[valid] - self.K_depth[0, 2]) * depth[valid] / self.K_depth[0, 0],
                (self.v_grid[valid] - self.K_depth[1, 2]) * depth[valid] / self.K_depth[1, 1],
                depth[valid],
            ],
            axis=1,
        )
        # Video cloud at the given resolution, with normals oriented towards the camera
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # Downsample before outlier removal for the sake of speed
        cloud = cloud.voxel_down_sample(self.voxel)
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=2 * self.voxel, max_nn=30)
        )
        # Cloud is in camera coordinates, so the camera sits at the origin
        cloud.orient_normals_towards_camera_location(np.zeros(3))
        return cloud
