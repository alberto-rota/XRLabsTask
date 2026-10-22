### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'registration.py' registers the video point cloud (camera RF) to the MRI skin point cloud (MRI RF), frame by frame.

Tracking strategy: every frame runs ICP starting from the previous pose (fast, precise, but local).
The global FPFH+RANSAC registration (slow, robust, coarse) only runs when tracking is lost, followed by ICP.
Every result (tracked or global) must pass a quality gate, otherwise the last good pose is kept.

> Needs the registration target from 'mri.py' and the registration source from 'surface_estimator.py'
> Provides the camera->MRI pose and the registration metrics
"""

import numpy as np
import open3d as o3d

class Registration:
    def __init__(
        self,
        target,
        voxel=2,
        max_patch_motion_mm=6,
        max_lost_frames=15,
        min_fitness_tracking=0.45,
        min_fitness_reinit=0.75,
    ):
        """Stores the MRI registration target and initializes the tracking state"""
        self.target = target  # Output of MRI.get_registration_target
        self.target_cloud = target["cloud"]
        self.voxel = voxel

        # REGISTRATION CHECKS
        # The registration is considered lost if the camera->MRI pose jumps too far away between frames.
        self.max_patch_motion_mm = max_patch_motion_mm
        self.max_lost_frames = max_lost_frames
        # There are also threshodlds on the registration fitness score 
        self.min_fitness_tracking = min_fitness_tracking
        self.min_fitness_reinit = min_fitness_reinit

        # Tracking state: last pose that passed the quality gate
        self.T_refined = None
        self.frames_since_good = 0  # grows while registration fails, widens the allowed motion

    # Geometric constraint
    @staticmethod
    def patch_motion(T_a, T_b, patch):
        """RMS distance (mm) the observed patch moves on the model between two camera->MRI poses."""
        D = T_a - T_b
        return np.sqrt(np.mean(np.sum((patch @ D[:3, :3].T + D[:3, 3]) ** 2, axis=1)))

    def ransac_near_pose(self, cloud, cloud_fpfh, T_prior, radius):
        """FPFH+RANSAC, with each video point only matched to MRI points within `radius` of where T_prior puts it.
        Keeps the re-initialization consistent with the motion prior (no flips on smooth, symmetric patches)."""
        predicted = np.asarray(cloud.points) @ T_prior[:3, :3].T + T_prior[:3, 3]
        feat = cloud_fpfh.data.T
        corr = [
            (i, idx[np.argmin(np.linalg.norm(self.target["feat"][idx] - feat[i], axis=1))])
            for i, idx in enumerate(self.target["tree"].query_ball_point(predicted, radius))
            if idx
        ]
        if len(corr) < 10:
            return T_prior
        return o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            cloud,
            self.target["cloud"],
            o3d.utility.Vector2iVector(np.array(corr, np.int32)),
            self.voxel,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
        ).transformation

    def ransac_global(self, cloud, cloud_fpfh):
        """FPFH+RANSAC over the whole MRI cloud, used when there is no prior pose"""
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=cloud,
            target=self.target["cloud"],
            source_feature=cloud_fpfh,
            target_feature=self.target["fpfh"],
            mutual_filter=True,
            max_correspondence_distance=1 * self.voxel,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    1 * self.voxel
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                400_000, 0.999
            ),
        ).transformation

    def find_rigid(self,source_cloud, T_init, ds=3.0):
        """Coarse-to-fine point-to-plane ICP from T_init. Returns the last ICP result."""
        for max_dist in (2 * ds, ds):
            result = o3d.pipelines.registration.registration_icp(
                source_cloud,
                self.target_cloud,
                max_dist,
                T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=ds)
                ),
            )
            T_init = result.transformation
        return result

    def register(self, cloud):
        """Runs the registration process on the video cloud, returns the final transformation and the metrics"""
        # Tracking: ICP from the previous pose, accepted only if the fit is good and the pose doesn't jump
        # Motion allowance: one frame of motion, plus one per lost frame (capped)
        self.frames_since_good += 1
        max_motion_mm = self.max_patch_motion_mm * min(self.frames_since_good, self.max_lost_frames)
        patch = np.asarray(cloud.points)[::10]

        def is_plausible(T_new):
            return self.T_refined is None or self.patch_motion(self.T_refined, T_new, patch) < max_motion_mm

        # Tracking: ICP from the previous pose, accepted only if the fit is good and the motion is plausible
        tracked = False
        if self.T_refined is not None:
            result = self.find_rigid(cloud, self.T_refined)
            tracked = fitness_gate(result, MIN_FITNESS=self.min_fitness_tracking) and is_plausible(result.transformation)

        # Tracking lost: FPFH+RANSAC, then ICP. Searched near the last pose if there is one, globally otherwise
        registered = tracked
        if not tracked:
            frame_fpfh = compute_fpfh_features(cloud, self.voxel)
            if self.T_refined is not None:
                T_init = self.ransac_near_pose(cloud, frame_fpfh, self.T_refined, radius=max_motion_mm + 5)
            else:
                T_init = self.ransac_global(cloud, frame_fpfh)
            result = self.find_rigid(cloud, T_init)
            registered = fitness_gate(result, MIN_FITNESS=self.min_fitness_reinit) and is_plausible(result.transformation)

        # Only a pose that passed the quality gate and the motion prior is kept. Otherwise the last good one is held
        patch_motion_mm = np.nan
        if registered:
            if self.T_refined is not None:
                patch_motion_mm = self.patch_motion(self.T_refined, result.transformation, patch)
            self.T_refined = result.transformation
            self.frames_since_good = 0

        return {
            "T_camera2mri": self.T_refined,
            "fitness": result.fitness,
            "inlier_rmse": result.inlier_rmse,
            "tracked": tracked,
            "registered": registered,
            "patch_motion_mm": patch_motion_mm,
            "num_points": len(cloud.points),
        }



### HELPERS ----------------------------------------------------------------

def compute_fpfh_features(
    cloud: o3d.geometry.PointCloud, ds: float = 3.0
) -> o3d.pipelines.registration.Feature:
    """FPFH descriptors (local 3D shape signatures), one per point of an already downsampled cloud.
    The cloud MUST carry oriented normals:
    """
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=6 * ds, max_nn=100)
    )

def fitness_gate(result, MIN_FITNESS=0.6, MAX_RMSE_mm=2.5):
    """Quality gate on an ICP result"""
    return result.fitness >= MIN_FITNESS and result.inlier_rmse <= MAX_RMSE_mm