import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker


def _seed_ts(seed, radius=10, **args):
    """Get seed time seriese."""
    if isinstance(seed, tuple):
        assert len(seed) == 3, "Coordinate must be in a 3D space"
        return NiftiSpheresMasker(seeds=[seed], radius=radius, **args)
    elif isinstance(seed, (str, nb.Nifti1Image)):
        return NiftiMasker(mask_img=seed, **args)
    else:
        raise ValueError("Seed must be a nifti image or a coordinate")


def _all_same(items):
    return all(x == items[0] for x in items)


def _scan_consistent(funcs):
    """Check all files has the same length and TR"""
    t_r = []
    n_scans = []
    for _, item in funcs.items():
        func = item["func"]
        example_func = nb.load(func)
        t_r.append(example_func.header.get_zooms()[-1])
        n_scans.append(example_func.shape[-1])
    if _all_same(t_r) and _all_same(n_scans):
        return t_r[0]
    else:
        raise ValueError(
            "Input functional images has different TRs "
            + "and different length of scan."
        )


def _get_frametime(t_r, n_scans):
    return np.linspace(0, (n_scans - 1) * t_r, n_scans)


def _seed_mat(seed_masker, func, confounds):
    ts = seed_masker.fit_transform(func)
    if ts.ndim == 2:
        ts = ts.mean(axis=1)
    ts = pd.DataFrame(ts, columns=["seed"])
    confounds = pd.read_csv(confounds, sep="\t")
    confounds = confounds.drop(columns=["constant"])
    dm = pd.concat([ts, confounds], axis=1)
    return dm


def _bulid_design(d, t_r):
    frametimes = _get_frametime(t_r, d.shape[0])
    design_matrix = make_first_level_design_matrix(
        frametimes,
        hrf_model="spm",
        add_regs=d.values,
        add_reg_names=d.columns.tolist(),
    )
    contrast = _first_level_seed_contrast(design_matrix.shape[1])
    return design_matrix, contrast


def _first_level_seed_contrast(design_mat_col):
    seed_contrast = np.array([1] + [0] * (design_mat_col - 1))
    return {"seed_based_glm": seed_contrast}


def _first_level(funcs, seed):
    """Run a first level seed base functional connectivity analysis."""
    t_r = _scan_consistent(funcs)
    seed_masker = _seed_ts(seed=seed)
    first_level_models = []
    first_level_contrasts = []
    for sub_id, item in funcs.items():
        func = item["func"]
        confounds = item["confounds"]
        d = _seed_mat(seed_masker, func, confounds)
        design_matrix, contrast = _bulid_design(d, t_r)
        model = FirstLevelModel(t_r=t_r, subject_label=sub_id)
        model = model.fit(run_imgs=func, design_matrices=design_matrix)
        first_level_models.append(model)
        first_level_contrasts.append(contrast)
    return first_level_models, first_level_contrasts


def _check_group_level(confounds, design_matrix, first_level_models):
    check_shape = _all_same(
        [confounds.shape[0], design_matrix.shape[0], len(first_level_models)]
    )
    if not check_shape:
        raise ValueError(
            "Number of first level results not matching second level design."
        )


def _check_group_design(design_matrix, contrast):
    col_dm = design_matrix.columns.tolist()[1:]
    col_ct = contrast.columns.tolist()
    if col_dm != col_ct:
        raise KeyError(
            "The header of group level design and contrast files"
            + " should be identical."
        )


def _group_level(
    first_level_models, group_confounds, group_design_matrix, group_contrast
):
    if isinstance(group_confounds, str):
        group_confounds = pd.read_csv(group_confounds)

    if isinstance(group_design_matrix, str):
        group_design_matrix = pd.read_csv(group_design_matrix)

    if isinstance(group_contrast, str):
        group_contrast = pd.read_csv(group_contrast)

    _check_group_level(group_confounds, group_design_matrix, first_level_models)
    _check_group_design(group_design_matrix, group_contrast)

    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(
        first_level_models, group_confounds, group_design_matrix
    )
    return second_level_model


# def report(first_level_models, second_level_model, contrast, output):
#     for model in first_level_models:
#         zmap = model.compute_contrast(contrast['seed_based_glm'],
#                                       output_type='z_score')
#         report = make_glm_report(model,
#                                 contrasts=contrast,
#                                 plot_type='glass',
#                                 )
#         report.save_as_html(output + 'first_level_report.html')

#     group_zmap = second_level_model.compute_contrast(
#                   first_level_contrast=contrast['seed_based_glm'])
