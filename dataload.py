# (c) Alberto for XRLabs - Interview task

import pydicom # Handling dicom
import numpy as np # Tensors
import matplotlib.pyplot as plt # Plotting
import os # File system
from rich import print # Logging (I like coloered output in the terminal)
from rich.progress import track # Progress bar
import argparse # CLI

from helpers import anatomical_orientation, find_transform_to_rpi

args = argparse.ArgumentParser()
args.add_argument("--data_dir", type=str, default="data/Aneurysm_MRI", help="path to dicom data dir, defaults to data/Aneurysm_MRI")
args = args.parse_args()
DATA_DIR = args.data_dir # For this task its the only arg

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

# Initializing volume 2 patient matrix as identity
vol2pat = np.eye(4)
# First col: moving down a row (row index) follows Y_dir (IOP[3:6]) * row spacing
vol2pat[:3,0] = Y_dir * pixel_spacing_Y
# Second col: moving along a row (column index) follows X_dir (IOP[0:3]) * column spacing
vol2pat[:3,1] = X_dir * pixel_spacing_X

# Third colums, slice spacings: computed from slice positions
imagepos_firstslice = np.array(slices[0].ImagePositionPatient, float)
imagepos_lastslice = np.array(slices[-1].ImagePositionPatient, float)
vol2pat[:3,2] = (imagepos_lastslice - imagepos_firstslice) / (len(slices) - 1)

# Fourth column transformss origin to imageposition for first slice
vol2pat[:3,3] = imagepos_firstslice
print("Volume-to-Patient transform matrix:")
print(vol2pat)

# Sanity check: voxel (0,0,k) must land on each slice ImagePositionPatient
print("Volume-to-Patient transform check: ", "[green]PASS[/green]" if all(np.allclose(vol2pat @ [0, 0, k, 1], [*np.array(slices[k].ImagePositionPatient, float), 1]) for k in (0, 1, len(slices) - 1)) else "[red]FAIL[/red]")

# Identifying anatomical orientation
anatomical_orientation_vol2pat = anatomical_orientation(vol2pat)

print(f"\nAnatomical orientation Identified: [blue]{anatomical_orientation_vol2pat}[/blue] ")
if anatomical_orientation != "RPI":
    vol2pat_rpi = vol2pat @ find_transform_to_rpi(vol2pat) 
print(f"Volume-to-RPI transform matrix:")
print(vol2pat_rpi)
anatomical_orientation_new = anatomical_orientation(vol2pat_rpi)
print(f"\nAnatomical orientation Identified: [blue]{anatomical_orientation_new}[/blue] --> [green]PASS[/green]" if anatomical_orientation_new == "RPI" else "[red]FAIL[/red]")

