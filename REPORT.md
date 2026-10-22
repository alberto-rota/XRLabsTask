# Alberto / XRLabs : Surgical Navigation Task - Report

## The problem 
The task boils down to one question asked at every frame of the video: where is the camera with respect to the MRI? If I can estimate `T_camera2mri` reliabely, everything else (the overlay, the metrics, the 3D view) follows from inverting it and projecting the MRI mesh onto the left frame. For this reason I treated the whole project as a chain that ends in that single transform, and I organzied the code around the two halves of the chain: Part A builds a clean target out of the MRI. Part B builds a point cloud out of the stereo video and aligns it to that target. Each module (`mri.py`, `surface_estimator.py`, `registration.py`, `logger.py`) owns one link of the chain, and `main.py` wires them together.

## Reference frames and registration target computation
Before writing any registration the MRI coordinate system needs to be made compliant. The DICOM came in PRI (I found), so I permuted and flipped it to RPI. On the video side I stuck with the OpenCV convention of the rectified left camera, and I made that the "depth camera" where the depth map, the point cloud and the overlay all live. The 1 voxel = 1mm spacing looked strange at first, but the metadata suggests the volume was already post-processed: on top of it, a feature-matching based registration would search for an affine transformation, which would include a scaling factor that could compensate for the mismatch in scale. 

For this MRI volume specifically, a simple thresholding operator would separate the body from the background: we apply a global threshold on the voxel intensity, keep the largest connected component, close and fill holes, then marching cubes. It is not robust to every sequences, but it is transparant, easy to debug and to inspect qualitatively to adjust the threshold and the smoothing parameter. The task asks for a 3D-to-3D match-based registration, and to this end we extract 3D features from the MRI volume (after converting it to a point-cloud-with-normals) with the FPFH pipeline

## Video processing and registration source compitation
For the video I rescaled the calibration to the squished side-by-side frames, rectifed, ran a stereo matcher and back-projected with the rectified intrinsics. The phatom surface is smooth and poorly texured, making the depthmap that we obtain noisy. A few assumptions help us obtain a better depthmap, among these: the phantom is the closest object, the depth lies between 500 and 1000mm, the camera motion between frames is small. As each frame pair becomes available in the video stream, we convert the obtained depthamp to a point cloud in the camera reference frame, which will act the the source cloud that is going to be registered on the target.

## Registration pipeline
We use two types of registration methods:
- **ICP**: Fast, only looks at geometry, requires a good enough initialization
- **RANSAC**: Slower, needs geometry and features, can be applied globally

At a given frame $k$, snce the camera move smoothly, ICP from the previous pose at frame $k-1$ is usually enough and it is fast. When such previous pose is unavailable (either at frame $k=0$ or when the ICP error is too high to adopt the obtained transformation with confidence), the pipeline falls back to RANSAC on the FPFH features, first near the previous pose and then globally. ICP still follows up to refine this transform.

Whether a registration transformation can be trusted depends on several factors: the quality gate on fitness value (from the `open3d` outputs), the inlier RMSE on the source and target cloud, the frame-to-frame discrepancy (with continuous motion we cannot allow for two consecutive depthmaps to be registed at very distant targetrs). 

Still, this decision process came from acepting upfront that the phantom does not perfectly match the MRI (the missing piece of cranium, the extra surface on the forehead) and that some frames would simply be hard, so the system needs to know when it is wrong.

## Evaluations
The metrics I report tells how well two surfaces fit and how stable the pose is, not how far a target inside the head is from where it is drawn. There is no ground truth and no annotated landmarks, so I chose not to claim a TRE or a reprojection error I could not compute. 

One approach might be a manual annnotation of 3D points of interest on the mesh, and a manual annotation of the same points on the video pixels.

Frames that fails the quality gate (around 2% of the total video) are still drawn for the sake of the demo, but marked with a red contour, since in a real deployement the right behavior would be to hide the overlay rather than show a confident-looking wrong one.

## Possible improvements for deploying in liv
For a production surgical-navigation system, the first thing I would change is how accuracy is defined and proven. Surface fitness and inlier RMSE are useful internal signals, but a surgeon cares about the target registration error at the structures that matter, so the system would have to be validated on phantoms and cadavers with fiducials or a tracked pointer, and report a TRE that is measured, not inferred. The registration target itself should also move closer to the anatomy of interest: segmenting the aneurysm, the vessels and the cortex from the MRI instead of just the skin, and once the craniotomy is open, registering to the bone or cortical surface that is actually exposed, possibly accounting for brain shift with a non-rigid model. On the input side, the calibratioDi  would need to follow the live zoom and focus of the microscope instead of being fixed at 401mm, and the heuristic depth band would be replaced by a proper segmentation of the anatomy that is robust to instruments and hands entering the field of view.

The second set of changes is about behaving safely and in real time. The pipeline would need to run at 30 FPS or more with bounded latency, with stereo matching and registration on the GPU and logging off the critical path, and a temporal filter (e.g. a Kalman filter) on the pose to remove jitter and bridge short failures. More importantly, the system has to fail visibly and conservatively: when the quality gate is not met the overlay should be hidden, not just outlined in red, and the confidence of the registration should be communicated to the surgeon at all times. Finally, all of this would need to live inside a medical-device software process, with traceable requirements, verification on a large set of recorded cases, and a risk analysis of every failure mode, because in the operating room a confident-looking wrong overlay is worse than no overlay at all.
