"""ITE model define.
"""

# Necessary modules
from treatments import CRN_Model, RMSN_Model


def treatment_effects_model(model_name, model_parameters, task):
    """Determine ITE model.

    Args:
        - model_name: 'CRN', 'RMSN'
        - model_parameters: parameters of the models
        - task: 'classification' or 'regression':

    Returns:
        - treatment_model: ITE model
    """
    assert model_name in ["CRN", "RMSN"]

    if model_name == "CRN":
        treatment_model = CRN_Model(task=task)

    elif model_name == "RMSN":
        treatment_model = RMSN_Model(task=task)

    treatment_model.set_params(**model_parameters)

    return treatment_model
