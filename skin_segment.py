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
from skimage.measure import marching_cubes  # Pretty standard algo for voxel-to-mesh
from skimage.measure import label  # Connected components
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter # Binary ops needed for mesh refinement

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
binary_body = np.array(vol3d_rpi > args.body_volume_threshold)
# Keep only the largest connected component (the head), dropping small noise blobs
components = label(binary_body)
binary_body = components == np.argmax(np.bincount(components.ravel())[1:]) + 1
# Close small 3D gaps, then fill the inside so that only the outer skin surface is meshed
# (meshing the raw intensities also produces internal tissue boundaries, bad for surface registration)
binary_body = binary_closing(binary_body, structure=np.ones((3, 3, 3)), iterations=2)
binary_body = binary_fill_holes(binary_body)

# Voxel to mesh -> Marching cubes. Light blurring of the mask removes the voxel staircase artifacts
verts, faces, normals, _ = marching_cubes(gaussian_filter(binary_body.astype(np.float32), sigma=1), level=0.5)
# Vertices have the 3D volume original coordinates so converting them to patient coordinates (mm)
verts_rpi = vox2pat_rpi@np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1).T
verts = verts_rpi.copy()[:3, :].T
# Normals are directions: only the rotation part applies
normals = normals @ vox2pat_rpi[:3, :3].T

### VISUALIZATION - Rerun viewer should open automatically ---------------------------------

# Setting up the rerun viewer
rr.init("skin_segmentation", spawn=True)
rr.send_blueprint(
    rr.blueprint.Horizontal(
        rr.blueprint.Spatial3DView(
            origin="/volume",
            name="volume",
        ),
        rr.blueprint.Spatial3DView(
            origin="/skin",
            name="skin",
        ),
    )
)
# Logging the volume on the left,
rr.log("/volume", rr.Volume3D(values=vol3d.astype(np.float16)))
# Loggin the skin mesh on the right
rr.log(
    "/skin",
    rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=faces.astype(np.uint32),
        vertex_normals=normals,
    ),
)

### SAVING OUTPUTS ------------------------------------------------------------
# Saving the skin mesh (vertices in patient mm, triangles, normals)
np.savez(os.path.join(args.out_dir, "skin_mesh.npz"), verts=verts, faces=faces, normals=normals)
print(f"> Skin mesh saved to skin_mesh.npz ({len(verts)} vertices, {len(faces)} triangles)")
