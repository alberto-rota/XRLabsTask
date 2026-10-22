### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'logger.py' handles the 3D visualization on Rerun, the processing metrics shown in the terminal and the
per-frame outputs saved to disk.

> Needs the outputs of 'mri.py', 'surface_estimator.py' and 'registration.py'
"""

import os
import numpy as np
import cv2
import json5 as json
import rerun as rr  # Crazy good 3D visualization library
from rerun.blueprint import EyeControls3D
import rerun.blueprint as rrb
from rich import print


class Logger:
    def __init__(self, out_dir="out/", save_frames=False, save_depthmaps=False, save_overlays=False, log_rectified=False, focussed_tab=0):
        """Opens the Rerun viewer with the blueprint for the MRI and the registration views"""
        self.out_dir = out_dir
        self.save_frames, self.save_depthmaps, self.save_overlays = save_frames, save_depthmaps, save_overlays
        self.log_rectified = log_rectified

        ### 3D VISUALIZATION ON RERUN -----------------------------------------------------------------
        # Setting up the visualization first. This includes the rerun viewer and its blueprint
        rr.init("Alberto - XRLabs Surgical Navigation Interview Task", spawn=True)

        # This sets up the whole panel arrangement in the rerun viewer for both of the parts.
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Tabs(
                    # Part A: MRI vol, SKin mesh, 2D slices
                    rrb.Vertical(
                        rrb.Horizontal(
                            rrb.Spatial3DView(
                                origin="/mri", contents=["/mri/vol_raw", "/mri/rpi_axes"], name="RAW MRI Volume",eye_controls=EyeControls3D(spin_speed=0.4)
                            ),
                            rrb.Spatial3DView(
                                origin="/mri", contents=["/mri/skin", "/mri/rpi_axes"], name="Skin Mesh",eye_controls=EyeControls3D(spin_speed=0.4)
                            ),
                            ),
                        rrb.Horizontal(
                            *[rrb.Spatial2DView(origin=f"/slide{i}") for i in range(5)]
                        ),
                        row_shares=[0.66,0.33],
                        name="Part A: MRI Preprocessing" 
                    ), 
                    # Part B: rectified frames and depth
                    rrb.Horizontal(
                        rrb.Vertical(
                            rrb.Horizontal(
                                rrb.Spatial2DView(
                                    origin="/rect/left", name="Left video feed (rectified)"
                                ),
                                rrb.Spatial2DView(
                                    origin="/rect/right", name="Right video feed(rectified)"
                                ),
                                
                            ),
                            rrb.Spatial2DView(
                                origin="/mri/depth_registered/camera/image", name="Depthmap (rectified left)"
                            ),
                            rrb.TimeSeriesView(origin="/errors", name="Errors"),
                            rrb.TimeSeriesView(origin="/metrics", name="Metrics"),
                            row_shares=[0.35,0.35,0.15,0.15]
                        ),
                        rrb.Vertical(
                            rrb.Spatial2DView(
                                origin="/overlay/left", name="Overlay (left)"
                            ),
                            rrb.Spatial3DView(
                                origin="/mri",
                                name="Registration",
                            ),
                        ),
                        name="Part B: Surgical Navigation Overlay",
                    ),
                    active_tab=focussed_tab,
                ),
                collapse_panels=True, 
            )
        )
        # Setting visualization orientation. LFD shows head up-right
        rr.log("/mri", rr.ViewCoordinates.LFD, static=True)

        # Progress bar in the terminal to show how far along we are in the video processing
        self.proc_metrics = {
            "avg_total_time_per_frame_ms": 0,
            "avg_stereo_matching_time_ms": 0,
            "avg_registration_time_ms": 0,
            "avg_logging_time_ms": 0,
            "total_registration_failures": 0,
        }
        # Per-frame registration history, saved to disk and summarized at the end
        self.history = {
            "frameidx": [],
            "T_camera2mri": [],
            "fitness": [],
            "inlier_rmse": [],
            "tracked": [],
            "registered": [],
            "patch_motion_mm": [],
            "num_points": [],
        }

    def log_head_mesh(self, vol3d_rpi, verts, faces, normals):
        """Logs the raw MRI volume and the skin mesh"""

        # Logging the volume on rerun: Rerun volumes should be (z-y-x), we have (x-y-z): permuting.
        rr.log(
            "/mri/vol_raw",
            rr.Volume3D(
                values=np.permute_dims(vol3d_rpi, axes=(2, 1, 0)).astype(np.float16),
                value_range=(0, 195),   # Manually set this because I like the viz it generates
                gamma=1.35,             # Manually set this because I like the viz it generates
                optical_density=13,     # Manually set this because I like the viz it generates
            ),
            static=True,
        )
        # Logging the segmetned skin/external surface as a mesh.
        rr.log(
            "/mri/skin",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces.astype(np.uint32),
                vertex_normals=normals,
                albedo_factor=[224, 172, 140, 255],  # skin tone
            ),
            static=True,
        )
        # RPI reference frame: arrows from the volume origin along increasing voxel index (x->Right, y->Posterior, z->Inferior)
        axis_length = 0.4 * max(vol3d_rpi.shape)
        rr.log(
            "/mri/rpi_axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * axis_length,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                labels=["R (+x)", "P (+y)", "I (+z)"],
                show_labels=True,
                radii=1.5,
            ),
            static=True,
        )
        # Logging 5 equally spaced slices
        for i in range(5): 
            rr.log(
                f"/slide{i}",
                rr.Image(vol3d_rpi[:, :, i * (vol3d_rpi.shape[2]-20)// 5+50]),
                static=True,
            )
        
    def log_rectification_guides(self, rect_size, line_spacing=60):
        """Drawing horizontal lines at 60px intervals in the rectified images to visually check stereo rectification"""
        rows = np.arange(0, rect_size[1], line_spacing)
        for view, width in (("/rect/left", rect_size[0]), ("/rect/right", rect_size[0])):
            guides = rr.LineStrips2D([[[0, y], [width, y]] for y in rows], colors=[0, 255, 0], radii=1)
            rr.log(f"{view}/guides", guides, static=True)

    def log_frames(self, frameidx, overlay, left_rect, right_rect, frame_viz_downsample=2):
        """Logs the overlay and the rectified pair for this frame"""
        rr.set_time("frame", sequence=frameidx)
        if overlay is not None:
            rr.log("/overlay/left", rr.Image(overlay[::frame_viz_downsample, ::frame_viz_downsample], color_model="BGR"))
        # Full resolution, so the epipolar guides (in rectified pixel coordinates) line up
        rr.log("/rect/left", rr.Image(left_rect, color_model="BGR"))
        rr.log("/rect/right", rr.Image(right_rect, color_model="BGR"))

    def log_depth_surface(self, depth, K_depth, rect_size, T_camera2mri):
        """Logs the depth map from the rectified left camera, placed in the MRI RF by the registration"""
        if T_camera2mri is not None:
            rr.log(
                "/mri/depth_registered",
                rr.Transform3D(
                    translation=T_camera2mri[:3, 3], mat3x3=T_camera2mri[:3, :3]
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
                resolution=rect_size,
                image_plane_distance=20,
            ),
        )
        rr.log(
            "/mri/depth_registered/camera/image",
            rr.DepthImage(depthviz, colormap="turbo",depth_range=depthvizrange)
        )

    def log_metrics(self, frameidx, reg):
        """Logs the registration metrics of this frame to Rerun and stores them in the history"""
        if not reg["registered"]:
            self.proc_metrics["total_registration_failures"] += 1
        rr.log("/metrics/fitness", rr.Scalars(reg["fitness"]))
        rr.log("/metrics/global_reinit", rr.Scalars(float(not reg["tracked"])))
        rr.log("/metrics/registered", rr.Scalars(float(reg["registered"])))
        rr.log("/errors/inlier_rmse", rr.Scalars(reg["inlier_rmse"]))
        if not np.isnan(reg["patch_motion_mm"]):
            rr.log("/errors/patch_motion_mm", rr.Scalars(reg["patch_motion_mm"]))

        self.history["frameidx"].append(frameidx)
        for key in list(self.history)[1:]:
            self.history[key].append(np.full((4, 4), np.nan) if key == "T_camera2mri" and reg[key] is None else reg[key])

    def update_mean(self, key, seconds, frameidx):
        """Running mean in ms over the processed frames"""
        self.proc_metrics[key] += (seconds * 1000 - self.proc_metrics[key]) / (frameidx+1)

    def metrics_description(self, frameidx):
        return (
            f"[cyan]Match: {self.proc_metrics['avg_stereo_matching_time_ms']:.0f}ms, "
            f"Registration: {self.proc_metrics['avg_registration_time_ms']:.0f}ms, "
            f"Logging: {self.proc_metrics['avg_logging_time_ms']:.0f}ms, "
            f"Total: {self.proc_metrics['avg_total_time_per_frame_ms']:.0f}ms, "
            f"FAILURES: {self.proc_metrics['total_registration_failures']}/{max(frameidx+1, 1)} ({self.proc_metrics['total_registration_failures'] / max(frameidx+1, 1):.0%})"
        )

    def save_frame_outputs(self, frameidx, left_rect, right_rect, depth, overlay):
        """Saving frames, depthmaps and overlays to disk if requested"""
        if self.save_frames:
            os.makedirs(os.path.join(self.out_dir, "left"), exist_ok=True)
            np.save(
                os.path.join(self.out_dir, "left", f"left_{frameidx:04d}.npy"),
                left_rect,
            )
            os.makedirs(os.path.join(self.out_dir, "right"), exist_ok=True)
            np.save(
                os.path.join(self.out_dir, "right", f"right_{frameidx:04d}.npy"),
                right_rect,
            )
        if self.save_depthmaps:
            os.makedirs(os.path.join(self.out_dir, "depthmaps"), exist_ok=True)
            np.save(
                os.path.join(
                    self.out_dir, "depthmaps", f"depth_{frameidx:04d}.npy"
                ),
                depth,
            )
        if self.save_overlays and overlay is not None:
            os.makedirs(os.path.join(self.out_dir, "overlays"), exist_ok=True)
            cv2.imwrite(
                os.path.join(self.out_dir, "overlays", f"overlay_{frameidx:04d}.png"),
                overlay,
            )

    def summary(self):
        """Checks the filesaves, prints the registration summary and saves the registration history to disk"""
        total_frames = len(self.history["frameidx"])
        # Checking filesaves
        for flag, folder, name in (
            (self.save_frames, "left", "frames"),
            (self.save_depthmaps, "depthmaps", "depth maps"),
            (self.save_overlays, "overlays", "overlays"),
        ):
            if flag:
                num_saved = len(os.listdir(os.path.join(self.out_dir, folder))) if os.path.isdir(os.path.join(self.out_dir, folder)) else 0
                save_ok = num_saved == total_frames
                print(
                    f"Saved {'[green]' if save_ok else '[red]'} {num_saved}/{total_frames} [/] {name} to {os.path.join(self.out_dir, folder)}"
                )
        if total_frames == 0:
            return

        ### REGISTRATION SUMMARY ------------------------------------------------------------
        registered = np.array(self.history["registered"])
        tracked = np.array(self.history["tracked"])
        fitness = np.array(self.history["fitness"])
        inlier_rmse = np.array(self.history["inlier_rmse"])
        patch_motion = np.array(self.history["patch_motion_mm"], dtype=float)
        registration_metrics = {
            "frames_processed": total_frames,
            "frames_registered": int(registered.sum()),
            "registration_failure_rate": float(1 - registered.mean()),
            "global_reinit_rate": float(1 - tracked.mean()),
            "fitness_mean": float(fitness[registered].mean()) if registered.any() else None,
            "fitness_median": float(np.median(fitness[registered])) if registered.any() else None,
            "inlier_rmse_mm_mean": float(inlier_rmse[registered].mean()) if registered.any() else None,
            "inlier_rmse_mm_median": float(np.median(inlier_rmse[registered])) if registered.any() else None,
            "inlier_rmse_mm_p95": float(np.percentile(inlier_rmse[registered], 95)) if registered.any() else None,
            "patch_motion_mm_median": float(np.nanmedian(patch_motion)) if not np.isnan(patch_motion).all() else None,
            "avg_times_ms": {k: v for k, v in self.proc_metrics.items() if k.startswith("avg")},
        }
        print("\nRegistration summary (metrics over the registered frames):")
        for key, value in registration_metrics.items():
            print(f"  {key}: {f'{value:.3f}' if isinstance(value, float) else value}")

        T_last = self.history["T_camera2mri"][-1]
        print("\nLast registered pose:")
        print(T_last)
        print("Direction: rectified left camera coordinates (mm) [blue]-->[/blue] MRI mesh coordinates. Invert it for the MRI --> camera overlay")

        with open(os.path.join(self.out_dir, "registration_metrics.json"), "w") as f:
            json.dump(registration_metrics, f, indent=4)
        print("> Registration metrics saved to registration_metrics.json")
        np.savez(
            os.path.join(self.out_dir, "registration.npz"),
            **{k: np.array(v) for k, v in self.history.items()},
        )
        print("> Per-frame poses and metrics saved to registration.npz")

    def export_video_overlay(self):
        """Exports the overlay images to a full-resolution mp4 video"""
        overlay_dir = os.path.join(self.out_dir, "overlays")
        if not os.path.isdir(overlay_dir):
            print(f"[red]No overlays found in {overlay_dir}, cannot export video[/red]")
            return
        overlay_files = sorted([f for f in os.listdir(overlay_dir) if f.endswith(".png")])
        if len(overlay_files) == 0:
            print(f"[red]No overlays found in {overlay_dir}, cannot export video[/red]")
            return
        first_frame = cv2.imread(os.path.join(overlay_dir, overlay_files[0]))
        height, width, _ = first_frame.shape
        out_video_path = os.path.join(self.out_dir, "overlay_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (width, height))
        for overlay_file in overlay_files:
            frame = cv2.imread(os.path.join(overlay_dir, overlay_file))
            out_video.write(frame)
        out_video.release()
        print(f"> Overlay video exported to {out_video_path}")