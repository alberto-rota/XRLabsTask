### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
'skin_segment.py' processes the 3D volume and generates the external surface of the body. It
includes a visualization tool to see the results in a 3D viewer.

'python skin_segment.py --help' to see the args overrides

> Needs the outputs of 'dataload.py'
> Provides outputs to 'register_depth.py'
"""

import numpy as np
import argparse
import json
import os
import rerun as rr  # Crazy good 3D visualization library
from rerun.blueprint import EyeControls3D
from skimage.measure import marching_cubes  # Pretty standard algo for voxel-to-mesh
from scipy.ndimage import (
    binary_closing,
    gaussian_filter,
)  # Binary ops needed for mesh refinement

args = argparse.ArgumentParser()
args.add_argument(
    "--vol_npy",
    type=str,
    default="out/volume.npy",
    help="path to npy tensor with 3D volume, defaults to 'out/volume.npy'",
)
args.add_argument(
    "--body_volume_threshold",
    type=int,
    default=30,
    help="threshold for body volume, defaults to 30",
)
args.add_argument(
    "--transforms_json",
    type=str,
    default="out/transforms.json",
    help="path to json file with transforms, defaults to 'out/transforms.json'",
)
args.add_argument(
    "--out_dir",
    type=str,
    default="out/",
    help="path ouput directory, defaults to out/",
)
args = args.parse_args()

### DATA LOADING ------------------------------------------------------------

# Loading volume from npy file
vol3d = np.load(args.vol_npy)
# Loading volume to RPI transformation matrix
with open(args.transforms_json, "r") as f:
    transforms = json.load(f)
vox2pat_rpi = np.array(transforms["vox2pat_rpi"])

if len(vol3d.shape) != 3:
    raise ValueError("Volume must be a 3D tensor")
vol3d_rpi = vol3d.copy().transpose(1, 0, 2)

### EXTERNAL SURFACE GENERATION ------------------------------------------------------------

# Generate initial binary mask by segmentation
mri_intensity_threshold = args.body_volume_threshold
vol_binary = np.array(vol3d_rpi > mri_intensity_threshold)
vol_filtered = gaussian_filter(vol3d_rpi.astype(np.float32), sigma=0.5)
field = np.where(
    vol_binary,
    np.maximum(vol_filtered, mri_intensity_threshold + 1),
    np.minimum(vol_filtered, mri_intensity_threshold - 1),
)
verts, faces, normals, _ = marching_cubes(field, level=mri_intensity_threshold)

print(vol_filtered.shape, vol_filtered.min(), vol_filtered.max())
rr.init("MRI Preprocessing", spawn=True)

rr.send_blueprint(
    rr.blueprint.Horizontal(
        rr.blueprint.Spatial3DView(
            origin="/mri", contents="/mri/vol_raw", name="RAW MRI Volume",eye_controls=EyeControls3D(spin_speed=0.4)
        ),
        rr.blueprint.Spatial3DView(
            origin="/mri", contents="/mri/skin", name="Skin Mesh",eye_controls=EyeControls3D(spin_speed=0.4)
        ),
    ),

)
# Voxel axes are RPI: X -> patient Right (viewer's Left when facing them), Y -> Posterior (into the
# screen), Z -> Inferior (down). LFD is right-handed like RPI, so nothing gets mirrored
rr.log("/mri", rr.ViewCoordinates.LFD, static=True)
# Logging the volume on rerun: Rerun volumes should be (z-y-x), we have (x-y-z): permuting.
rr.log(
    "/mri/vol_raw",
    rr.Volume3D(
        values=np.permute_dims(vol3d_rpi, axes=(2, 1, 0)).astype(np.float16),
        value_range=(0, 195),   # Manually set this because I like the viz it generates
        gamma=1.35,             # Manually set this because I like the viz it generates
        optical_density=13,     # Manually set this because I like the viz it generates
    ),
)
rr.log(
    "/mri/skin",
    rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=faces.astype(np.uint32),
        vertex_normals=normals,
    ),
)

### SAVING OUTPUTS ------------------------------------------------------------
# Saving the skin mesh (vertices in patient mm, triangles, normals)
np.savez(
    os.path.join(args.out_dir, "skin_mesh.npz"),
    verts=verts,
    faces=faces,
    normals=normals,
)
print(
    f"> Skin mesh saved to skin_mesh.npz ({len(verts)} vertices, {len(faces)} triangles)"
)
