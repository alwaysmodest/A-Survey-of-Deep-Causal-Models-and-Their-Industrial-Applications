"""Preprocess the data to the proper format.

(1) One hot encoding (one_hot_encoding_module)
(2) Normalization (normalization_module)
(3) Data division (data_division_module)
(4) Temporal data formatting (temporal_data_formatting)

preprocess_parameters:
- problem: 'one-shot' or 'online'
- max_seq_len: maximum sequence length after padding
- label_name: the column name for the label(s)
- treatment: the column name for treatments
- normalization: feature normalization
- one_hot_encoding: input features that need to be one-hot encoded

Returns:
- training and testing datasets
    - temporal: temporal data
    - static: static data
    - label: label information
    - treatment: treatment data
    - time: measurement time information

- feature_name: temporal and static feature names
- normalization_parameters: MinMaxScaler parameters for recovering
"""

# Necessary packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_utils import list_diff, normalization, padding


def one_hot_encoding_module(x, s, one_hot_encoding_features):
    """One hot encoding for selected features.

    @deprecated: OneHotEncoder

    Args:
        - x: temporal features
        - s: static features
        - one_hot_encoding_features: list of features for one hot encoding

    Returns:
        - x: temporal features after one-hot encoding
        - s: static features after one-hot encoding
    """

    if len(one_hot_encoding_features) > 0:

        for feature_name in one_hot_encoding_features:
            if feature_name in x.columns:
                x = pd.get_dummies(x, columns=[feature_name])
            elif feature_name in s.columns:
                s = pd.get_dummies(s, columns=[feature_name])

    return x, s


def normalization_module(x, s):
    """Normalized selected features.

    @deprecated: MinMaxNormalizer

    Args:
        - x: temporal features
        - s: static features

    Returns:
        - x: temporal features after normalization
        - s: static features after normalization
        - norm_parameters: normalization parametres for renormalization
    """

    temporal_col_names = x.drop(["id", "time"], axis=1).columns.values
    static_col_names = s.drop(["id"], axis=1).columns.values

    x[temporal_col_names], temporal_norm_parameters = normalization(  # pylint: disable=no-value-for-parameter
        x[temporal_col_names]
    )

    s[static_col_names], static_norm_parameters = normalization(  # pylint: disable=no-value-for-parameter
        s[static_col_names]
    )

    norm_parameters = {
        "temporal_max_val": temporal_norm_parameters["max_val"],
        "temporal_min_val": temporal_norm_parameters["min_val"],
        "static_max_val": static_norm_parameters["max_val"],
        "static_min_val": static_norm_parameters["min_val"],
    }

    return x, s, norm_parameters


def data_division_module(x, s, preprocess_parameters):
    """Divide data into temporal, static, label, treatment, and time information.

    Args:
        - x: temporal features
        - s: static features
        - preprocess_parameters: problem, treatment, label_name are used in this function.

    Returns:
        - dataset: includes temporal, static, label, treatment, and time information.
    """

    problem = preprocess_parameters["problem"]
    treatment_features = preprocess_parameters["treatment"]
    label_name = preprocess_parameters["label_name"]

    # 1. Label define

    # For temporal labels
    if problem == "online":
        y = x[["id"] + label_name]
        x = x.drop(label_name, axis=1)
    # For static labels
    elif problem == "one-shot":
        y = s[label_name]
        s = s.drop(label_name, axis=1)

    # 2. Time define
    time = x[["id", "time"]]
    x = x.drop(["time"], axis=1)

    # 3. Treatment define
    if treatment_features == "None":
        treatment = np.zeros([0])
    else:
        if treatment_features[0] in x.columns:
            treatment = x[["id"] + treatment_features]
            x = x.drop(treatment_features, axis=1)
        elif treatment_features in s.columns:
            treatment = s[["id"] + treatment_features]
            s = s.drop(treatment_features, axis=1)

    s = s.drop(["id"], axis=1)

    dataset = {"temporal": x, "static": s, "label": y, "treatment": treatment, "time": time}

    return dataset


def temporal_data_formatting(x, max_seq_len):
    """Returns numpy array for predictor model training and testing.

    Args:
        - x: temporal data
        - max_seq_len: maximum sequence length

    Returns:
        - x_hat: preprocessed temporal data
    """

    uniq_id = np.unique(x["id"])

    x_hat = list()

    for i in tqdm(range(len(uniq_id))):

        idx_x = x.index[x["id"] == uniq_id[i]]

        if len(idx_x) >= max_seq_len:
            temp_x = x.loc[idx_x[-max_seq_len:]].drop(["id"], axis=1)
        else:
            temp_x = padding(x.loc[idx_x].drop(["id"], axis=1), max_seq_len)

        x_hat = x_hat + [np.asarray(temp_x)]

    x_hat = np.asarray(x_hat)

    return x_hat


def data_preprocess(train_x, train_s, test_x, test_s, preprocess_parameters):
    """Preprocess the data.
    (1) One-hot encoding
    (2) Normalization
    (3) Data division
    (4) Temporal data formatting

    Args:
        - train_x: training temporal data
        - train_s: training static data
        - test_x: testing temporal data
        - test_s: testing static data

        - preprocess_parameters:
            - problem: 'one-shot' or 'online'
            - max_seq_len: maximum sequence length after padding
            - label_name: the column name for the label(s)
            - treatment: the column name for treatments
            - normalization: feature normalization
            - one_hot_encoding: features that needs one hot encoding

    Returns:
        - train_data: training dataset
            - temporal: temporal data
            - static: static data
            - label: label information
            - treatment: treatment data
            - time: measurement time information

        - test_data: testing dataset
        - feature_name: temporal and static feature names
        - normalization_parameters: MinMaxScaler parameters for recovering
    """

    # 1. One hot encoding
    one_hot_encoding_features = preprocess_parameters["one_hot_encoding"]
    train_x, train_s = one_hot_encoding_module(train_x, train_s, one_hot_encoding_features)
    test_x, test_s = one_hot_encoding_module(test_x, test_s, one_hot_encoding_features)

    # 2. Normalize the features
    normalization = preprocess_parameters["normalization"]  # pylint: disable=redefined-outer-name

    if normalization:
        train_x, train_s, normalization_parameters = normalization_module(train_x, train_s)
        test_x, test_s, _ = normalization_module(test_x, test_s)

    # 2-1. Feature name for interpretation models
    temporal_features = list_diff(train_x.columns.values.tolist(), ["id", "time"])
    static_features = list_diff(train_s.columns.values.tolist(), ["id"])
    feature_name = {"temporal": temporal_features, "static": static_features}

    # 3. Observation, Treatment, Time, Label define
    train_data = data_division_module(train_x, train_s, preprocess_parameters)
    test_data = data_division_module(test_x, test_s, preprocess_parameters)

    # 4. Temporal data formatting
    set_length = train_data["temporal"].shape[0]
    max_seq_len = preprocess_parameters["max_seq_len"]

    for dict_name in train_data.keys():
        if train_data[dict_name].shape[0] == set_length:
            train_data[dict_name] = temporal_data_formatting(train_data[dict_name], max_seq_len)
            test_data[dict_name] = temporal_data_formatting(test_data[dict_name], max_seq_len)
        else:
            train_data[dict_name] = np.asarray(train_data[dict_name])
            test_data[dict_name] = np.asarray(test_data[dict_name])

    return train_data, test_data, feature_name, normalization_parameters
