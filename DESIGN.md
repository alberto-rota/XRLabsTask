# Alberto / XRLabs : Surgical Navigation Task - Design

**Contents**
- [Codebase structure](#codebase-structure)
- [Coordinate systems](#coordinate-systems)
- [Transformation directions](#transformation-directions)
- [Data Flow](#data-flow)
- [Key assumptions](#key-assumptions)
- [Known limitations](#known-limitations)
- [Room For Improvement](#room-for-improvement)

## Codebase structure

| File | Function |
| --- | --- |
| `main.py` | Main entrypoint, run the pipelines and calls the modules, each described below
| `mri.py` | DICOM loading, metadata checks, voxel-to-patient and RPI transforms, skin mesh extraction, registration target, overlay rendering |
| `surface_estimator.py` | Stereo video processing: splits the left-right frame, rectifies, runs stereo matching, depth map and point cloud |
| `registration.py` | Registration of the video point cloud to the MRI skin (FPFH + RANSAC + ICP, tracking and quality gates) |
| `logger.py` | Rerun visualization, metrics, and everything that gets saved to disk |

## Coordinate systems
| RF | Description | Axes |
| --- | --- | --- |
| **Voxel (original)** | Indices `(row, col, slice)` of the volume as it is stacked from the .dcm files | PRI for this dataset |
| **Voxel (RPI)** | Same volume with the axes permuted/flipped so that `+i` = Right, `+j` = Posterior, `+k` = Inferior. See rerun for a 3D view| RPI, 1 voxel = 1mm (strange) |
| **Camera (rectified left)** | The "depth camera": left camera after stereo rectification. The depth map, the video point cloud and the overlay all live here | OpenCV: +x right, +y down, +z forward |
| **Image (rectified left)** | Pixels of the rectified left frame | `u` right, `v` down |
| **3D Viewer** | Rerun uses an LPS coordinate system (just for visualization). At some point we set the transform once with `rr.ViewCoordinates` | LPS (+x Left, +y Posterior, +z Superior) |

> [!IMPORTANT]
> It's **very unusual** that the voxel spacing and slice spacing is 1 voxel/mm, but I see from the metadata that the DICOM has been parsed with ITK or other post processing sofrware so it makes sense. Also, because the 3D-to-3D feature-based transformation with **RANSAC will be affine and therefore will solve the scale inconsistency** with the metric depth map.

## Transformation directions

| Transform | Where | From → To |
| --- | --- | --- |
| `vox2pat` | `mri.py` | Voxel (original) → Patient (LPS) |
| `to_rpi` | `mri.py` | Voxel (RPI) → Voxel (original) |
| `vox2pat_rpi = vox2pat @ to_rpi` | `mri.py` | Voxel (RPI) → Patient (LPS) |
| `K_depth` (= `P1[:3,:3]`) | `surface_estimator.py` | Camera (rect. left) → Image (rect. left) |
| `R1`, `P1` + distortion (rectification maps) | `surface_estimator.py` | Raw left image → Rectified left image |
| `T_camera2mri` | `registration.py` | Camera (rect. left) → Voxel (RPI) |
| `T_mri2camera = inv(T_camera2mri)` | `main.py` | Voxel (RPI) → Camera (rect. left) |

Ultimately, with each incoming frame from the video feed we want to estimate `T_camera2mri` which will align the point cloud from the depthmap to the MRI. The overlay needs the inverse because the MRI mesh vertices must be brought to the camera RF in order to be projected to the overlay.

$$ p_{img} \sim K_{depth} \cdot [I | 0] \cdot T_{camera2mri}^{-1} \cdot p_{RPI} $$

and if we want it in patient coordinates instead, $p_{LPS} = T_{vox2pat\_rpi} \cdot p_{RPI}$.

## Data Flow
```
                       PART A                                              PART B (per frame)
  .dcm slices                                                SBS video frame
      │  sort by InstanceNumber, metadata checks                  │  split left/right
      ▼                                                           ▼
  vol3d + vox2pat ──► orientation (PRI) ──► to_rpi          rectify (K rescaled, R1/R2, P1/P2)
      │                                       │                   │
      ▼                                       ▼                   ▼
  vol3d_rpi ◄──────────────────────── vox2pat_rpi            SGBM disparity ──► depth = fB / d
      │  threshold, largest component,                            │  foreground band (40mm)
      │  closing, fill holes, marching cubes                      ▼
      ▼                                                       back-project with K_depth
  skin mesh (RPI, mm)                                             │  2mm voxel, outlier removal, normals
      │  sample 100k pts, 2mm voxel, outward normals              ▼
      ▼                                                       video cloud (camera RF)
  MRI cloud + FPFH + KD-tree  ─────────────► Registration ◄───────┘
                                                 │  ICP tracking from previous pose
                                                 │  └ lost? FPFH + RANSAC (near prior or global) + ICP
                                                 │  quality gate (fitness, RMSE, motion)
                                                 ▼
                                            T_camera2mri ──► invert ──► raycast mesh with K_depth ──► overlay
                                                 │
                                                 ▼
                                         Rerun + metrics + disk
```
(I drew this with the help of an LLM)

## Key assumptions
**DICOM file preprocessing**
- All slices have the same voxel resolution and uniform slice spacing (checked), so the slice step is (last position − first position) / (N − 1)
- The volume axes are close to the patient axes. In practice the axis values for the volume RF and the RPI RF should have the largest entry (in absolute value) corresponding to the axis index. In practice, an axis like `[0.86, 0.02, -0.11]` is be considered as aligned to `[1,0,0]` so +X. An oblique acquisition this still gives the closest RPI, but the "RPI" volume will not be exactly axis aligned

**Skin mesh computation**
- The head is the largest connected component above the intensity threshold (30 by default, with max value 255)
- Rigid anatomy: the phantom does not deform, so the MRI surface is the same surface the camera sees

**Surface estimation from video**
- The calibration (done at 1920x1080 per camera) is valid for the video once fx and cx are rescaled by 954/1920 (the SBS video squishes each eye horizontally, y is untouched)
- The calibration matches the focus/zoom of the video (the params are given at 401mm focus)
- The phantom is the object closest to the camera: everything deeper than 40mm behind the nearest surface is removed as background
- **Depth range is between 500 and 1000mm**: this is the default range that I set epirically and by looking at the video. Can be overridden with `--depth_range_mm`

**Registration**
- The motion between consecutive frames is small, the camera motion is continuous and smooth (≤6mm of patch motion per frame): this is also used as a consistency check frame by frame.


## Known limitations
The main limitation is that **the phantom in the video is not a perfect correspondence to the MRI data**, as there is a missing portion of the cranium with exposed brain matter on the right side of the patient head and there is an additional curved surface on the phantom forehead. Still, random-sampling-consensus and ICP can deal with this but in some frames where these inconsistencies are visible registration can fail. 
Moreover:

- **No anatomical ground truth.** The errors I report (fitness, inlier RMSE, frame-to-frame motion) say how well the surfaces fit and how stable the pose is, not how far a target inside the head is from where we draw it (TRE). A surface fit can be very good and still be slid along a smooth patch of the head
- **No reprojection error.** The overlay is rendered from the registered pose but I don't measure a 2D error on it, as there are no annotated 2D landmarks to compare against
- **Background removal is a heuristic.** If an instrument or a hand gets closer to the camera than the phantom, the 40mm band selects the wrong thing
- **Small field of view.** The camera only sees a small patch of the head at ~0.5-1m with a ~19mm baseline; on flat-ish patches the registration is weakly constrained in the in-plane directions
- **Thresholding for the skin.** One global intensity threshold. It works for this MRI specifically but it's not robust to bias field or different sequences
- **Not real time.** ~100ms per frame (~10 FPS) on my machine (CPU only computation, MacBook Pro M4), and ~40ms of that is logging to Rerun and can be ignored

> [!IMPORTANT]
> On average, on 2% of the total frames registration fails, as the depth-to-MRI transformation cannot be considered reliable. Qualitatively, it happens often when the video shows the right side of the phantom (the brain tissue has not target on the cranium mesh) or if a large portion of the background table is in view.

In this case, for the sake of the demo, we still draw the overlay on the video and the low-reliability is represented as a thick red contour. On a real deployment, we would stand down and not display the overlay on low-confidence registrations.

## Room For Improvement
- **Validation with ground truth**: fiducials or a tracked pointer on a phantom to measure TRE at targets, plus a reprojection error on annotated landmarks in both views
- **Real anatomical targets**: render the structures that matter (the aneurysm, vessels, cortex) segmented from the MRI, and register to the surface that is actually exposed (bone, cortex) once the craniotomy is open
- **Temporal filtering**: a Kalman filter on the pose to smooth jitter and to predict through short failures, instead of holding the last pose
- **Performance**: run the stereo and registration on GPU, move logging to another async thread, target ≥30 FPS with bounded latency
