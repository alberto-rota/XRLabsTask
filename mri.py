### Alberto / XRLabs Interview task
# ------------------------------------------------------------------------------
"""
The MRI module loads the DICOM volume data and the associated metadata, computes the necessary transforms
to convert the volume to the RPI convention, extracts the skin surface mesh, prepares the registration
target and renders the mesh overlay from a given camera pose.
"""

import pydicom # Handling dicom
import numpy as np # Tensors
import os # File system
from rich import print # Logging (I like coloered output in the terminal)
from rich.progress import track # Progress bar
from skimage.measure import marching_cubes  # voxel-to-mesh
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    label,
    gaussian_filter,
)  # Binary ops needed for mesh refinement
from scipy.spatial import cKDTree
import open3d as o3d

import json5 as json
from registration import compute_fpfh_features
import cv2


class MRI:
    def __init__(self, data_dir):
        """Loads dicom data and metadata, computes the voxel-to-patient transform and the conversion to RPI

        data_dir: folder with the .dcm slices (ordered by InstanceNumber when available, by filename otherwise)
        """

        # Retrieving all filenames with dcm extensions and sorting base on filename
        dicom_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.dcm')]
        dicom_files.sort()
        # Actually loading the data
        slices = []
        for f in track(dicom_files, description="Loading DICOM slices..."):
            slices.append(pydicom.dcmread(os.path.join(data_dir, f)))
        print(f"Found {len(slices)} .dcm files in {data_dir}")

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
        self.vol3d = np.stack([s.pixel_array for s in slices], axis=2) # Stacking as (row,col,slice)
        print(f"\nVolume shape (voxels): {self.vol3d.shape}")

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

        # Original DICOM spatial metadata, printed and saved alongside the transforms
        self.metadata = {
            "ImageOrientationPatient": [float(v) for v in image_orientation_patient],
            "ImagePositionPatient_first": imagepos_firstslice.tolist(),
            "ImagePositionPatient_last": imagepos_lastslice.tolist(),
            "PixelSpacing": [pixel_spacing_X, pixel_spacing_Y],
            "SliceThickness": float(slices[0].SliceThickness),
            "SliceSpacing_from_positions": float(np.linalg.norm(vox2pat[:3,2])),
            "SliceDirection": Z_dir.tolist(),
            "NumSlices": len(slices),
            "SliceOrder": "InstanceNumber" if slices_have_idx else "filename",
        }
        print("\nOriginal DICOM spatial metadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")

        print("\nVolume-to-Patient transform matrix:")
        print(vox2pat)

        # CONVERTING-TO-RPI ---------------------------------------------------------------
        # Identifying anatomical orientation
        self.orientation = anatomical_orientation(vox2pat)

        print(f"\nAnatomical orientation Identified: [blue]{self.orientation}[/blue] ")
        # If not in RPI convention, converting to RPI
        if self.orientation != "RPI":
            to_rpi = find_transform_to_rpi(vox2pat)
            print("Transform required to express the volume in RPI (right-multiplied to Volume-to-Patient):")
            print(to_rpi)
        else:
            to_rpi = np.eye(4)
            print("No transformation required, already in RPI convention")
        vox2pat_rpi = vox2pat @ to_rpi
        print(f"Volume-to-RPI transform matrix:")
        print(vox2pat_rpi)
        # Verifying that the final transform is in RPI convention
        anatomical_orientation_new = anatomical_orientation(vox2pat_rpi)
        print(f"\nAnatomical orientation Identified: [blue]{anatomical_orientation_new}[/blue] --> " + ("[green]PASS[/green]" if anatomical_orientation_new == "RPI" else "[red]FAIL[/red]"))

        self.vox2pat, self.to_rpi, self.vox2pat_rpi = vox2pat, to_rpi, vox2pat_rpi

    def save(self, out_dir):
        """Saves transforms, metadata and raw volume to disk

        out_dir: destination folder, gets transforms.json and volume.npy
        """
        # Saving transformations to json
        with open(os.path.join(out_dir, "transforms.json"), "w") as f:
            json.dump(
                {
                    "vox2pat": self.vox2pat.tolist(),
                    "to_rpi": self.to_rpi.tolist(),
                    "vox2pat_rpi": self.vox2pat_rpi.tolist(),
                    "orientation": self.orientation,
                    "metadata": self.metadata,
                }, f, indent=4
            )
        print("> Transformations saved to transforms.json")
        np.save(os.path.join(out_dir, "volume.npy"), self.vol3d)
        print("> Volume saved to volume.npy")

    def estimate_skin_surface(self, body_volume_threshold=30):
        """Computes the skin surface mesh from the dicom volume

        Thresholds the RPI volume, keeps the largest connected component, closes and fills it, then runs marching
        cubes on the smoothed intensities. Also builds the raycasting scene used by render_from_pose.

        body_volume_threshold: MRI intensity separating tissue from background (also the marching cubes iso-level)

        Returns (verts, faces, normals) of the skin mesh, in RPI voxel coordinates
        """

        # vox2pat is basically a pure rotaion
        self.vol3d_rpi = self.vol3d.copy().transpose(1, 0, 2)

        # Generate initial binary mask by segmentation
        mri_intensity_threshold = body_volume_threshold
        # Rough binary segmentation based on what is tissue and what is backgroundf
        vol_binary = np.array(self.vol3d_rpi > mri_intensity_threshold)
        # Labelling the binary mask
        vol_labels = label(vol_binary)[0]
        counts = np.bincount(vol_labels.ravel())
        counts[0] = 0 # Background is not a body part

        # 3D binary operations to refine the mask: 1. binary closing to fill holes, 2. binary fill holes to fill small gaps
        vol_binary = binary_fill_holes(binary_closing(vol_labels == counts.argmax(), iterations=2))
        # Gaussian filter otherwise the mesh has strong staircase artifacts
        vol_filtered = gaussian_filter(self.vol3d_rpi.astype(np.float32), sigma=0.5)
        # Creating a field based on the binary mask and the filtered volume
        field = np.where(
            vol_binary,
            np.maximum(vol_filtered, mri_intensity_threshold + 1),
            np.minimum(vol_filtered, mri_intensity_threshold - 1),
        )
        # Apply marching cubes
        self.verts, self.faces, self.normals, _ = marching_cubes(field, level=mri_intensity_threshold)
        print(f"\nExtracted external skin surface mesh with {len(self.verts)} vertices and {len(self.faces)} triangles")

        self.mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.verts),
            o3d.utility.Vector3iVector(self.faces),
        )
        # Computing normals for the head mesh
        self.mesh.compute_vertex_normals()

        # Preparing the scene to render the overlay of the MRI mesh that will be composed onto the video frame
        self.overlay_scene = o3d.t.geometry.RaycastingScene()
        self.overlay_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        return self.verts, self.faces, self.normals

    def save_skin_mesh(self, out_dir):
        """Saves the skin mesh (vertices, triangles, normals) to disk

        out_dir: destination folder, gets skin_mesh.npz
        """
        skin_mesh_path = os.path.join(out_dir, "skin_mesh.npz")
        np.savez(
            skin_mesh_path,
            verts=self.verts,
            faces=self.faces,
            normals=self.normals,
        )
        print(
            f"> Skin mesh saved to '{skin_mesh_path}' ({len(self.verts)} vertices, {len(self.faces)} triangles)"
        )

    def get_registration_target(self, cloud_size=100_000, voxel=2):
        """Returns the point cloud and features that we will use to register the camera to the MRI

        cloud_size: number of points sampled uniformly on the skin mesh
        voxel: downsampling resolution in mm, must match the video cloud so the FPFH descriptors are comparable

        Returns a dict with the downsampled cloud (outward normals), its FPFH features (as the Open3D object and as an
        (N, 33) array) and a KD-tree over the cloud points for fast radius/nearest neighbor queries
        """
        # MRI mesh to point cloud of the desired resolution
        mri_pcloud = self.mesh.sample_points_uniformly(
            number_of_points=cloud_size, use_triangle_normal=True
        )
        mri_points = np.asarray(mri_pcloud.points)
        mri_normals = np.asarray(mri_pcloud.normals)
        print(f"MRI mesh sampled as point cloud")
        print(
            f"{len(mri_points)} vertices & {len(mri_normals)} normals: {'[green]PASSED[/]' if len(mri_points) == len(mri_normals) else '[red]FAIL[/]'}"
        )

        # Make sure normals point outward (the normal direction should on  average be the same as the centerofmass-to-point direction)
        if np.mean(np.sum((mri_points - mri_points.mean(axis=0)) * mri_normals, axis=1)) < 0:
            print("Head mesh normals are inward, flipping")
            mri_normals = -mri_normals
            mri_pcloud.normals = o3d.utility.Vector3dVector(mri_normals)

        # Downsample the point cloud for the sake of speed
        cloud = mri_pcloud.voxel_down_sample(voxel)
        cloud.normalize_normals()

        # Computing FPH features
        fpfh = compute_fpfh_features(cloud, voxel)
        print(
            f"Computed FPFH features from the MRI head mesh, with shape {fpfh.data.shape}"
        )
        # Function returns the cloud itself, the feature object with their metadata and a KDTree to speed up the nearest neighbor search during registration (KDTree was Claude suggestion)
        return {
            "cloud": cloud,
            "fpfh": fpfh,
            "feat": fpfh.data.T,
            "tree": cKDTree(np.asarray(cloud.points)),
        }

    def render_from_pose(
        self,
        img,
        K_depth,
        T_mri2camera,
        rect_size=(1920, 1080),
        color=(140, 172, 224),
        alpha=0.5,
        outline_color=None,
    ):
        """Renders the shaded MRI mesh from a camera pose and alpha-composites it onto the video frame

        img: BGR frame from the rectified left camera
        K_depth: intrinsics of the rectified left camera
        T_mri2camera: 4x4 pose bringing MRI coordinates into the camera frame (inverse of the registration output)
        rect_size: (width, height) of the rendered image, must match img
        color: BGR color of the mesh
        alpha: opacity of the mesh over the frame
        outline_color: if given, draws the mesh silhouette in this BGR color (e.g. to flag a failed registration)

        Returns the composited BGR image

        NOTE: An LLM implemented this function for me. None of this code is mine
        """

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K_depth),
            o3d.core.Tensor(T_mri2camera),
            rect_size[0],
            rect_size[1],
        )
        hit = self.overlay_scene.cast_rays(rays)
        mask = np.isfinite(hit["t_hit"].numpy())
        # Headlight (Lambert) shading: surfaces turned away from the camera get darker -> contours show up
        dirs = rays.numpy()[..., 3:]
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        shade = np.clip(
            np.abs(np.sum(hit["primitive_normals"].numpy() * dirs, axis=-1)), 0.25, 1.0
        )

        overlay = img.copy()
        rendered = shade[mask, None] * np.array(color, dtype=np.float32)[None, :]
        overlay[mask] = (alpha * rendered + (1 - alpha) * img[mask]).astype(np.uint8)
        if outline_color is not None:
            # Silhouette: outer border of the rendered mesh
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(overlay, contours, -1, outline_color, 2, cv2.LINE_AA)
        return overlay

### HELPERS ----------------------------------------------------------------

def anatomical_orientation(vox2pat: np.ndarray) -> str:
    """Determines the anatomical orientation of the volume based on the voxel-to-patient transform

    For each column/axis, we assume that the largest absolute value identifies which of the cartesian direction is the
    axis mostly aligned with, while its sign identifies the axis direction.

    vox2pat: 4x4 voxel-to-patient transform (patient coordinates in DICOM LPS)

    Returns the 3-letter orientation code (e.g. "RPI"), one letter per voxel axis
    """
    # Positive-Negative axis have this labels, will be appended to the string based on the axis dirs found
    positive_labels = {0: "L", 1: "P", 2: "S"}
    negative_labels = {0: "R", 1: "A", 2: "I"}
    anatomical_orientation = ""

    for colidx, column in enumerate(vox2pat.T[:3]):
        # What row in each column has the max absoute value? (most of axis direction is aligned here)
        main_axis = np.argmax(np.abs(column[:3]))
        # Appending the label dependin on axis dir
        is_val_positive = column[main_axis] > 0
        anatomical_orientation += (
            positive_labels[main_axis]
            if is_val_positive
            else negative_labels[main_axis]
        )
    return anatomical_orientation


def find_transform_to_rpi(vox2pat: np.ndarray) -> np.ndarray:
    """Computes the transform that converts the volume from its current anatomical orientation to RPI

    vox2pat: 4x4 voxel-to-patient transform (patient coordinates in DICOM LPS)

    Returns the 4x4 transform to right-multiply to vox2pat
    """
    # create the transform that represents the RPI reference frame ()
    rpi_rf = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # Then I instantiate the PRI reference directions (vox2pat rotation part)
    pri_rf = np.eye(4)
    pri_rf[:3, :3] = vox2pat[:3, :3]

    # Computing the relative transform from PRI to RPI
    pri_to_rpi = np.linalg.inv(pri_rf) @ rpi_rf
    return pri_to_rpi