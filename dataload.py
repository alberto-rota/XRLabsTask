import pydicom # Handling dicom
import numpy as np # Tensors
import matplotlib.pyplot as plt # Plotting
import os # File system
from rich import print # Logging (I like coloered output)

import argparse
# Input data as provided
DATA_DIR = "data/Aneurysm_MRI"
# Retrieving all filenames
dicom_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.dcm')]
dicom_files.sort()

# Loading each slice with dcmread in list
slices = [pydicom.dcmread(os.path.join(DATA_DIR, f)) for f in dicom_files]
# Do all .dcm file have an instance number?
slices_have_idx = all([s.InstanceNumber is not None for s in slices])
print(f"InstanceNumber metadata present in all files: {slices_have_idx}")

# Reordering based on slice idx
if slices_have_idx:
    slices.sort(key=lambda s: int(s.InstanceNumber)) # metadata is string

# List to tensor (accessign pixel_array key)
vol3d = np.stack([s.pixel_array for s in slices]) 
D,H,W = vol3d.shape # Depth - height - width
print("MRI shape:", vol3d.shape) 
