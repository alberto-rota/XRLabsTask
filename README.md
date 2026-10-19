# Alberto / XRLabs : Task

### Environment set up
Get the code first 
```bash
git clone 
```

I use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage my environments. After installation, in the repo that you cloned:
```bash
uv sync
```
It reads the `pyproject.toml` file, solves depencencies, creates a virtual environment and installs. 

To run the script, activate the virual environment with
```bash
source .venv/bin/activate
```

> [!NOTE]
> If you want to use plain `pip`, just `pip install -r requirements.txt` in your environment

### Part A
`dataload.py` deals with dataloading, parses the metdata and computes the transformmations to the RPi convention requested by the task. Run it with 

```bash
python dataload.py --data_dir <PATH_TO_DIR_WITH_DCM>
```

##### Usage of the DICOM metadata and computation of voxel2patient transformation
The script loads the .dcm files in the folder and checks the following:
1. All files have the Position, Orientation, Thickness and Spacing metadata parameters that we expect 
2. These parameters are consistent across slices
3. All files have a valid InstanceNumber

If that is the case, the InstanceNumber param is used as an ordering parameter to stack the slices in a 3D tensor along the last (third) dimension.

The `ImageOrientationPatient` parameter is split into the X and Y direction (first 3 and last 3 values). This pair of 3D-vectors makes up the first two columns of the voxel-to-patient transformation matrix (`vox2pat` in the script). Because of the requested convention (row-col-slice), the Y direction will be the first column and the X direction will be the second.
The third column of `vox2pat` is computed as the difference in the `ImagePositionPatient` parameter of the first and last slice in the stack. 

The translation component of `vox2pat` is assigned to the `ImagePositionpatient` of the first slice, so that the origin of the refernce frame coincides with slide 0.

##### Transforming to RPI
`helpers.py` contains a function that identifies what direction is represented by which column axis and retrieves the three-letter anatomical convention with a lookup. If it identifies anything different to RPI, it transforms it.

The transformation to RPI is based on the following logic:
1. The RPI direction vectors for the row-col-slice conventions are known (`rpi_rf` in the file)
2. The transformation of `vox2mat` to the RPI reference frame can be obtained as their relative pose (actually only the relative rotation), and because they are both homogeneous the transformation is simply $T_{vox2mat}^{-1}T_{RPI}$
3. The final trasnformation from volume to RPI is the post-multiplication

We check the obtained anatomical convention obtained with the same helper function as earlier and verify that its an RPI.

