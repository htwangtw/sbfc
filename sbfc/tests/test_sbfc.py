import os
from nilearn import datasets
import pandas as pd
import numpy as np
from ..parser import seed_base_connectivity


mask = os.path.dirname(__file__) + "/data/difumo64_pcc.nii.gz"


def _make_data():
    adhd_dataset = datasets.fetch_adhd(n_subjects=2)
    group_confounds = pd.DataFrame(adhd_dataset.phenotypic)[
        ["Subject", "MeanFD", "age", "sex"]
    ]
    group_confounds = group_confounds.rename(
        columns={"Subject": "subject_label"}
    )
    group_design_matrix = pd.DataFrame(adhd_dataset.phenotypic)[["Subject"]]
    group_design_matrix = group_design_matrix.rename(
        columns={"Subject": "subject_label"}
    )
    group_design_matrix["pheno"] = np.random.rand(2)
    group_contrast = pd.DataFrame([1], columns=["pheno"])

    # Prepare timing
    # t_r = 2.0
    # n_scans = 176
    func_img = {
        f"{sub_id}": (func, confound)
        for func, confound, sub_id in zip(
            adhd_dataset.func, adhd_dataset.confounds, group_confounds.index
        )
    }
    return func_img, group_design_matrix, group_confounds, group_contrast


def test_sbfc():
    (
        func_img,
        group_design_matrix,
        group_confounds,
        group_contrast,
    ) = _make_data()
    # Prepare seed
    pcc_coords = (0, -53, 26)
    first_m, first_con, s_m = seed_base_connectivity(
        func_img,
        pcc_coords,
        group_confounds,
        group_design_matrix,
        group_contrast,
    )
    assert len(first_m) == 2

    # mask seed
    first_m, first_con, s_m = seed_base_connectivity(
        func_img, mask, group_confounds, group_design_matrix, group_contrast
    )
    assert len(first_m) == 2
