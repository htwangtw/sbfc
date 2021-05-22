from . import model as sm


def seed_base_connectivity(
    data,
    seed,
    group_confounds,
    group_design_matrix,
    group_contrast,
    mask_img=None,
    drift_model="cosine",
    hrf_model=None,
    write_dir=None,
    verbose=0,
):
    """Perform seed-based functional connectivity analysis on resting state data.

    Parameters
    ----------

    data: dict
        A dictionary containing subject ID as key and point to the
        path to functional data and confound regressors.
        Structure should be like:
        data = {"subject_id": {
                    "func": ["path/to/func1.nii.gz", "path/to/func2.nii.gz"],
                    "confound": ["path/to/conf1.nii.gz", "path/to/conf2.nii.gz"]
                    }
                }

    seed: str or nifti image or tuple
        A 3D roi mask or oordinate of the seed coordinate.

    group_confounds: str or pandas.DataFrame
        A file or path to file  containing confound variables for group level analysis.
        The first roll must be headers.
        The first column mus have header 'subject_label',
        containing subject ID identical to `func`.

    group_design_matrix: str or pandas.DataFrame
        A file or path to file containing design matrix for group level analysis.
        The first roll must be headers.
        The first column mus have header 'subject_label',
        containing subject ID identical to `func`.

    group_contrast: str or pandas.DataFrame
        A file or path to file containing contrast for group level analysis.
        The first roll must be headers.
        The headers must be identical with regressor names in the design matrix.
        Each following roll is a contrast specification

    """
    subject_level_models = []
    for sub_id, items in data.items():
        subject_lvl, subject_lvl_contrast = sm.subject_level(
            seed,
            items["func"],
            confounds=items["confound"],
            subject_label=sub_id,
            write_dir=write_dir,
            verbose=verbose,
            drift_model=drift_model,
            hrf_model=hrf_model,
        )
        subject_level_models.append(subject_lvl)

    second_level_model = sm.group_level(
        subject_level_models,
        group_confounds,
        group_design_matrix,
        group_contrast,
        mask_img=mask_img,
        verbose=verbose,
    )
    return subject_level_models, subject_lvl_contrast, second_level_model
