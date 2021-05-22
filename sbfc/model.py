from os import getcwd, mkdir, path

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker
from nilearn.reporting import make_glm_report


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
    """Check a list of functional files files has the same length and TR"""
    t_r = []
    n_scans = []
    for func in funcs:
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


def _seed_mat(seed_masker, func, confounds=None):
    if confounds is not None:
        confounds = pd.read_csv(confounds, sep="\t")
        confounds = confounds.drop(columns=["constant"])
    ts = seed_masker.fit_transform(func, confounds=confounds)
    ts = ts.mean(axis=1)
    ts_m = ts.mean()
    ts_sd = ts.std()
    ts -= ts_m
    ts /= ts_sd
    ts = pd.DataFrame(ts, columns=["seed"])
    if isinstance(confounds, pd.DataFrame):
        return pd.concat([ts, confounds], axis=1)
    else:
        return ts


def _bulid_design(d, t_r, **args):
    frametimes = _get_frametime(t_r, d.shape[0])
    design_matrix = make_first_level_design_matrix(
        frametimes,
        add_regs=d.values,
        add_reg_names=d.columns.tolist(),
        **args,
    )
    contrast = _first_level_seed_contrast(design_matrix.shape[1])
    return design_matrix, contrast


def _first_level_seed_contrast(design_mat_col):
    seed_contrast = np.array([1] + [0] * (design_mat_col - 1))
    return {"seed": seed_contrast}


def subject_level(
    seed, funcs, confounds=None, subject_label=None, write_dir=None, verbose=0, **args
):
    """Run a subject level seed base functional connectivity analysis.
    One run - normal first level
    More than one run - retrun fixed effect of the seed.
    """
    hrf_model = args.get("hrf_model", False)
    if "hrf_model" in args:
        args.pop("hrf_model")
    if not isinstance(hrf_model, str):
        hrf_model = None

    mask_img = args.get("mask_img", False)
    if "mask_img" in args:
        args.pop("mask_img")
    if not isinstance(mask_img, str):
        mask_img = None

    t_r = _scan_consistent(funcs)
    seed_masker = _seed_ts(seed=seed)
    if confounds is None:
        confounds = [None] * len(funcs)

    # make output dir
    if write_dir is None:
        write_dir = path.join(getcwd(), f"results/{subject_label}/")
    if not path.exists(write_dir):
        mkdir(write_dir)

    design = []
    print("Generate design")
    for func, conf in zip(funcs, confounds):
        sm = _seed_mat(seed_masker, func, conf)
        # all contrast should be the same
        design_matrix, contrast = _bulid_design(sm, t_r, hrf_model=hrf_model, **args)
        design.append(design_matrix)

    contrast_id = "seed"
    print("Fit model")
    model = FirstLevelModel(
        t_r=t_r,
        smoothing_fwhm=5.0,
        hrf_model=hrf_model,
        subject_label=subject_label,
        mask_img=mask_img,
        verbose=verbose,
    )
    model = model.fit(run_imgs=funcs, design_matrices=design)

    print("Computing contrasts and save to disc...")
    statsmaps = model.compute_contrast(contrast[contrast_id], output_type="all")
    for map in statsmaps:
        image_path = path.join(write_dir, f"{contrast_id}_{map}.nii.gz")
        statsmaps[map].to_filename(image_path)

        # subject_label, map_name and effects_map_path
    report = make_glm_report(
        model,
        contrasts=contrast,
        plot_type="glass",
    )
    report_path = path.join(write_dir, "first_level_report.html")
    report.save_as_html(report_path)
    return model, contrast


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


def group_level(
    first_level_models,
    group_confounds,
    group_design_matrix,
    group_contrast,
    mask_img,
    verbose=0,
):
    if isinstance(group_confounds, str):
        group_confounds = pd.read_csv(group_confounds)

    if isinstance(group_design_matrix, str):
        group_design_matrix = pd.read_csv(group_design_matrix)

    if isinstance(group_contrast, str):
        group_contrast = pd.read_csv(group_contrast)

    _check_group_level(group_confounds, group_design_matrix, first_level_models)
    _check_group_design(group_design_matrix, group_contrast)

    second_level_model = SecondLevelModel(mask_img=mask_img, verbose=verbose)
    second_level_model = second_level_model.fit(
        first_level_models, group_confounds, group_design_matrix
    )
    return second_level_model
