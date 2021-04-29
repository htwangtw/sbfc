from . import model as sm


def seed_base_connectivity(
    funcs, seed, group_confounds, group_design_matrix, group_contrast
):
    first_level_models, first_level_contrasts = sm._first_level(funcs, seed)
    second_level_model = sm._group_level(
        first_level_models,
        group_confounds,
        group_design_matrix,
        group_contrast,
    )
    return first_level_models, first_level_contrasts, second_level_model
