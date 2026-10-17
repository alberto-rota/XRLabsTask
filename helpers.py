# (c) Alberto for XRLabs - Interview task

import numpy as np

def anatomical_orientation(vol2pat: np.ndarray) -> str:
    # Positive-Negative axis have this labels, will be appended to the string based on the axis dirs found
    positive_labels = {0:'L',1:'P',2:'S'}
    negative_labels = {0:'R',1:'A',2:'I'}
    anatomical_orientation = ""

    for colidx, column in enumerate(vol2pat.T[:3]):
        # What row in each column has the max absoute value? (most of axis direction is aligned here)
        main_axis = np.argmax(np.abs(column[:3])) 
        # Appending the label dependin on axis dir
        is_val_positive = column[main_axis] > 0
        anatomical_orientation += positive_labels[main_axis] if is_val_positive else negative_labels[main_axis]
    return anatomical_orientation

def find_transform_to_rpi(vol2pat: np.ndarray) -> np.ndarray:
    # create the transform that represents the RPI reference frame ()
    rpi_rf = np.array([ [-1,  0,  0,  0],
                        [ 0,  1,  0,  0],
                        [ 0,  0, -1,  0],
                        [ 0,  0,  0,  1]])
    # Then I instantiate the PRI reference directions (vol2pat rotation part)
    pri_rf = np.eye(4)
    pri_rf[:3,:3] = vol2pat[:3,:3]

    # Computing the relative transform from PRI to RPI
    pri_to_rpi = np.linalg.inv(pri_rf) @ rpi_rf
    return pri_to_rpi
