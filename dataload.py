### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
    'dataload.py' loads the DICOM volume data and the associated metadata then computes the necessary transforms
    to convert the volume to the RPI convention.
    
    'python dataload.py --help' to see the args overrides

    > Needs the raw DICOM data
    > Provides outputs to 'skin_segment.py'
"""

import pydicom # Handling dicom
import numpy as np # Tensors
import os # File system
from rich import print # Logging (I like coloered output in the terminal)
from rich.progress import track # Progress bar
import argparse # CLI

import json
from helpers import anatomical_orientation, find_transform_to_rpi

args = argparse.ArgumentParser()
args.add_argument("--data_dir", type=str, default="data/Aneurysm_MRI", help="path to dicom data dir, defaults to data/Aneurysm_MRI")
args.add_argument("--out_dir", type=str, default="out/", help="path outpiut directory, defaults to out/")
args = args.parse_args()
DATA_DIR = args.data_dir 

print()
### DATALOADING --------------------------------------------------

# Retrieving all filenames with dcm extensions and sorting base on filename
dicom_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.dcm')]
dicom_files.sort()
# Actually loading the data
slices = []
for f in track(dicom_files, description="Loading DICOM slices..."):
    slices.append(pydicom.dcmread(os.path.join(DATA_DIR, f)))
print(f"Found {len(slices)} .dcm files in {DATA_DIR}")

# Do all .dcm file have an instance number?
slices_have_idx = all([s.InstanceNumber is not None for s in slices])
print("InstanceNumber metadata check: ", "[green]PASS[/green]" if slices_have_idx else "[red]FAIL[/red]")
# Reordering based on slice idx (I trust this more than the filename)
if slices_have_idx:
    slices.sort(key=lambda s: int(s.InstanceNumber)) # metadata is string

# ImageOrientationPatient: checking that it is unique in all slices 
set_of_unique_orientations = np.unique(np.array([s.ImageOrientationPatient for s in slices]), axis=0)
print("Orientation check: ", "[green]PASS[/green]" if len(set_of_unique_orientations) == 1 else "[red]FAIL[/red]")
set_of_unique_spacings = np.unique(np.array([s.PixelSpacing for s in slices]), axis=0)
print("Slice Spacing check: ", "[green]PASS[/green]" if len(set_of_unique_spacings) == 1 else "[red]FAIL[/red]")
set_of_unique_thicknesses= np.unique([s.SliceThickness for s in slices])
print("Slice Thickness check: ", "[green]PASS[/green]" if len(set_of_unique_thicknesses) == 1 else "[red]FAIL[/red]")
image_orientation_patient = slices[0].ImageOrientationPatient
X_dir,Y_dir = np.array(image_orientation_patient[:3],float), np.array(image_orientation_patient[3:6],float)
Z_dir = np.cross(X_dir, Y_dir) # Right-handed, computing Z direction
pixel_spacing_X = float(slices[0].PixelSpacing[0])
pixel_spacing_Y = float(slices[0].PixelSpacing[1])

# List to tensor (accessign pixel_array key)
vol3d = np.stack([s.pixel_array for s in slices], axis=2) # Stacking as (row,col,slice)
H,W,D = vol3d.shape
print(f"\nVolume shape (voxels): {vol3d.shape}") 

### TRANSFORM: Volume-to-Patient --------------------------------------------------

# Initializing the voxel2patient matrix as identity
vox2pat = np.eye(4)
# First col: moving down a row (row index) follows Y_dir in 'pixel_spacing_Y' steps
vox2pat[:3,0] = Y_dir * pixel_spacing_Y
# Second col: moving along a row (column index) follows X_dir in 'pixel_spacing_X' steps
vox2pat[:3,1] = X_dir * pixel_spacing_X

# Third colums, slice spacings: computed from slice positions
imagepos_firstslice = np.array(slices[0].ImagePositionPatient, float)
imagepos_lastslice = np.array(slices[-1].ImagePositionPatient, float)
vox2pat[:3,2] = (imagepos_lastslice - imagepos_firstslice) / (len(slices) - 1)

# Fourth column transformss origin to imageposition for first slice
vox2pat[:3,3] = imagepos_firstslice
print("Volume-to-Patient transform matrix:")
print(vox2pat)

# Sanity check: voxel (0,0,k) must land on each slice ImagePositionPatient
print("Volume-to-Patient transform check: ", "[green]PASS[/green]" if all(np.allclose(vox2pat @ [0, 0, k, 1], [*np.array(slices[k].ImagePositionPatient, float), 1]) for k in (0, 1, len(slices) - 1)) else "[red]FAIL[/red]")

# CONVERTING-TO-RPI ---------------------------------------------------------------

# Identifying anatomical orientation
anatomical_orientation_vox2pat = anatomical_orientation(vox2pat)

print(f"\nAnatomical orientation Identified: [blue]{anatomical_orientation_vox2pat}[/blue] ")
# If not in RPI convention, converting to RPI
if anatomical_orientation != "RPI":
    vox2pat_rpi = vox2pat @ find_transform_to_rpi(vox2pat) 
else:
    vox2pat_rpi = vox2pat 
    print("No transformation required, already in RPI convention")
print(f"Volume-to-RPI transform matrix:")
print(vox2pat_rpi)
# Verifying that the final transform is in RPI convention
anatomical_orientation_new = anatomical_orientation(vox2pat_rpi)
print(f"\nAnatomical orientation Identified: [blue]{anatomical_orientation_new}[/blue] --> [green]PASS[/green]" if anatomical_orientation_new == "RPI" else "[red]FAIL[/red]")

# Saving transformations to json
with open(os.path.join(args.out_dir, "transforms.json"), "w") as f:
    json.dump(
        {
            "vox2pat": vox2pat.tolist(),
            "vox2pat_rpi": vox2pat_rpi.tolist(),
        }, f, indent=4
    )
print("> Transformations saved to transforms.json")
np.save(os.path.join(args.out_dir, "volume.npy"), vol3d)
print("> Volume saved to volume.npy")