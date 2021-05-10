from . import model as sm


def seed_base_connectivity(
    funcs, seed, group_confounds, group_design_matrix, group_contrast
):
    """Perform seed-based functional connectivity analysis on resting state data.

    Parameters
    ----------

    funcs: dict
        A dictionary containing subject ID as key and point to the
        path to functional data and confound regressors.
        Structure should be like:
        data = {"subject_id": {
                    "func": "path/to/func.nii.gz",
                    "confounds": "path/to/confounds.tsv"
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
    first_level_models, first_level_contrasts = sm._first_level(funcs, seed)
    second_level_model = sm._group_level(
        first_level_models,
        group_confounds,
        group_design_matrix,
        group_contrast,
    )
    return first_level_models, first_level_contrasts, second_level_model
