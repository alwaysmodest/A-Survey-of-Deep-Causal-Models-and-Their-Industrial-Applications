"""Normalization module.

Jinsung Note: I made separate class for MinMaxScaler and StandardScaler
              even though they are similar (due to the following sklearn module)
              I also added renormalization (ReScaler) module as well.
"""
from base import BaseEstimator, DataPreprocessorMixin


def normalizer(df, feature_names, normalizer_type):
    """Normalizer.

    Args:
        - df: input data
        - feature_names: features for normalization
        - normalizer_type: minmax or standard

    Returns:
        - df: normalized data
        - norm_parameters: parameters for renomalization
    """
    for f in feature_names:
        assert f in df.columns
    assert normalizer_type in ["standard", "minmax"]

    if normalizer_type == "standard":
        subtract_val = df[feature_names].mean()
        division_val = df[feature_names].std()
    elif normalizer_type == "minmax":
        subtract_val = df[feature_names].min()
        division_val = df[feature_names].max() - df[feature_names].min()

    for col_name in feature_names:
        df[col_name] = df[col_name] - subtract_val[col_name]
        df[col_name] = df[col_name] / (division_val[col_name] + 1e-8)

    norm_parameters = {"subtract_val": subtract_val, "division_val": division_val}

    return df, norm_parameters


def renormalizer(df, norm_parameters):
    """Renormalizer.

    Args:
        - df: input data
        - norm_parameters: parameters for renomalization

    Returns:
        - df: renormalized data
    """
    subtract_val = norm_parameters["subtract_val"]
    division_val = norm_parameters["division_val"]

    feature_names = subtract_val.keys()

    for f in feature_names:
        assert f in df.columns

    for col_name in feature_names:
        df[col_name] = df[col_name] * (division_val[col_name] + 1e-8)
        df[col_name] = df[col_name] + subtract_val[col_name]

    return df


class MixMaxScaler(BaseEstimator, DataPreprocessorMixin):
    def __init__(self, temporal_feature_names, static_feature_names):
        self.temporal_feature_names = temporal_feature_names
        self.static_feature_names = static_feature_names

    def fit_transform(self, dataset):
        if len(self.temporal_feature_names) > 0:
            if dataset.temporal_data is not None:
                dataset.temporal_data, temporal_norm_parameters = normalizer(
                    dataset.temporal_data, self.temporal_feature_names, normalizer_type="minmax"
                )
        if len(self.static_feature_names) > 0:
            if dataset.static_data is not None:
                dataset.static_data, static_norm_parameters = normalizer(
                    dataset.static_data, self.static_feature_names, normalizer_type="minmax"
                )
        norm_parameters = {
            "normalizer": "minmaxscaler",
            "temporal": temporal_norm_parameters,
            "static": static_norm_parameters,
        }

        return dataset, norm_parameters


class StandardScaler(BaseEstimator, DataPreprocessorMixin):
    def __init__(self, temporal_feature_names, static_feature_names):
        self.temporal_feature_names = temporal_feature_names
        self.static_feature_names = static_feature_names

    def fit_transform(self, dataset):
        if len(self.temporal_feature_names) > 0:
            if dataset.temporal_data is not None:
                dataset.temporal_data, temporal_norm_parameters = normalizer(
                    dataset.temporal_data, self.temporal_feature_names, normalizer_type="standard"
                )
        if len(self.static_feature_names) > 0:
            if dataset.static_data is not None:
                dataset.static_data, static_norm_parameters = normalizer(
                    dataset.static_data, self.static_feature_names, normalizer_type="standard"
                )
        norm_parameters = {
            "normalizer": "standardscaler",
            "temporal": temporal_norm_parameters,
            "static": static_norm_parameters,
        }

        return dataset, norm_parameters


class ReScaler(BaseEstimator, DataPreprocessorMixin):
    def __init__(self, norm_parameters):
        self.temporal_norm_parameters = norm_parameters["temporal"]
        self.static_norm_parameters = norm_parameters["static"]

    def fit_transform(self, dataset):
        if dataset.temporal_data is not None and self.temporal_norm_parameters is not None:
            dataset.temporal_data = renormalizer(dataset.temporal_data, self.temporal_norm_parameters)
        if dataset.static_data is not None and self.static_norm_parameters is not None:
            if dataset.static_data is not None:
                dataset.static_data = renormalizer(dataset.static_data, self.static_norm_parameters)

        return dataset
