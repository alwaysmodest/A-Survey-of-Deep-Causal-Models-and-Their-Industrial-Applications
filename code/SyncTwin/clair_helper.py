import logging
import os

import numpy as np
import pandas as pds
import tensorflow as tf
import torch

# from config import DEVICE
from util import io_utils


def get_clair_data(seed, sim_id, fold):

    base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
    data_path = base_path_data + "/{}-{}.{}"

    # loading config and data
    print("loading data")
    (
        n_units,
        n_treated,
        n_units_total,
        step,
        train_step,
        control_sample,
        noise,
        n_basis,
        n_cluster,
    ) = io_utils.load_config(data_path, fold)
    (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        m,
        sd,
        treatment_effect,
    ) = io_utils.load_tensor(data_path, fold)
    y_full_complete = torch.load(data_path.format(fold, "y_full_all", "pth"))
    ts_complete = torch.cat([x_full, y_full_complete], dim=0)
    time_mat = torch.arange(ts_complete.shape[0])[:, None, None].repeat(1, ts_complete.shape[1], 1).to(ts_complete)

    id_mat = torch.arange(ts_complete.shape[1])[None, :, None].repeat(ts_complete.shape[0], 1, 1).to(ts_complete)

    treat_mat = (time_mat > x_full.shape[0]).to(ts_complete)
    treat_mat[:, :n_units, :] = 0.0

    # id, time, treatment, X1, X2, Outcome

    ts_complete = torch.cat([id_mat, time_mat, treat_mat, ts_complete], dim=-1)
    mat = (
        ts_complete.permute((1, 0, 2))
        .reshape(ts_complete.shape[0] * ts_complete.shape[1], ts_complete.shape[2])
        .cpu()
        .numpy()
    )

    df = pds.DataFrame(data=mat, columns=["id", "time", "treatment", "x1", "x2", "outcome"])
    df["x3"] = df["outcome"]

    mat_static = np.stack([np.arange(ts_complete.shape[1]), np.zeros(ts_complete.shape[1])], axis=-1)
    df_static = pds.DataFrame(data=mat_static, columns=["id", "x_static"])
    max_seq_len = x_full.shape[0] + y_full.shape[0]
    projection_horizon = y_full.shape[0]

    return df, df_static, max_seq_len, projection_horizon, n_units, n_units_total, treatment_effect


def tf_set_log_level(level: int) -> None:
    if level >= logging.FATAL:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    if level >= logging.ERROR:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if level >= logging.WARNING:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logging.getLogger("tensorflow").setLevel(level)


def silence_tf() -> None:
    tf_set_log_level(logging.FATAL)
