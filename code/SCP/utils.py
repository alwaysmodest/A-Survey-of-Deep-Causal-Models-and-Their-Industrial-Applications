from os import listdir
from os.path import isfile, join
from typing import NamedTuple

import numpy as np
import torch
from ignite.handlers import Checkpoint

from dr_crn import DR_CRN
from mlp import NN_SCP


def bootstrap_RMSE(err_sq):
    rmse_list = []
    for i in range(500):
        new_err = err_sq[torch.randint(len(err_sq), err_sq.shape)]
        rmse_itr = torch.sqrt(torch.mean(new_err))
        rmse_list.append(rmse_itr.item())
    return np.std(np.array(rmse_list))


class HyperParameterConfig(NamedTuple):
    itr: int
    n_confounder_rep: int
    n_outcome_rep: int
    mmd_sigma: float
    lam_factual: float
    lam_propensity: float
    lam_mmd: float
    learning_rate: float
    batch_size: int


def check_n_hidden(model_path, model_id):
    file_list = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    model_file = [f for f in file_list if f.find(model_id) >= 0]
    if len(model_file) > 1:
        print("Multiple files exist: taking the first one.")
    model_file = model_file[0]

    checkpoint = torch.load(model_path + model_file)
    n_hidden = checkpoint["mlp.0.bias"].shape[0]
    return n_hidden


def load_model(model, model_path, model_id):
    file_list = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    model_file = [f for f in file_list if f.find(model_id) >= 0]
    if len(model_file) > 1:
        print("Multiple files exist: taking the first one.")
    model_file = model_file[0]

    checkpoint = torch.load(model_path + model_file)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    print("Model loaded from ", model_file)
    return model, model_file


def create_DR_CRN(single_cause_index, d_config, param, linear_model=False):
    model = DR_CRN(
        single_cause_index,
        d_config.n_confounder,
        d_config.n_cause,
        d_config.n_outcome,
        param.n_confounder_rep,
        param.n_outcome_rep,
        param.mmd_sigma,
        param.lam_factual,
        param.lam_propensity,
        param.lam_mmd,
        linear=linear_model,
    )
    return model


def create_NN_SCP(single_cause_index, d_config, param):
    model = NN_SCP(
        single_cause_index,
        d_config.n_confounder,
        d_config.n_cause,
        d_config.n_outcome,
        param.n_confounder_rep,
        param.n_outcome_rep,
        param.mmd_sigma,
        param.lam_factual,
        param.lam_propensity,
        param.lam_mmd,
    )
    return model


def get_scp_config(n_itr, n_confounder, p_confounder_cause):
    param_list = list()

    for i in range(n_itr):
        config = HyperParameterConfig(
            itr=i,
            n_confounder_rep=np.random.randint(int(n_confounder * p_confounder_cause), n_confounder + 1),
            n_outcome_rep=np.random.randint(int(n_confounder * (1 - p_confounder_cause)) + 1, n_confounder + 1),
            mmd_sigma=1.0,
            lam_factual=1.0,
            lam_propensity=1.0,
            # lam_mmd=np.random.choice([0.,  1.], size=1)[0],
            lam_mmd=1.0,
            # learning_rate=np.random.choice([0.001,  0.005,  0.01], size=1)[0],
            batch_size=int(np.random.choice([50, 100, 200], size=1)[0]),
            learning_rate=0.005,
            # batch_size=100,
        )
        param_list.append(config)

    return param_list


def get_nn_config_three(n_itr, n_confounder, n_cause):
    param_list = list()

    for i in range(n_itr):
        config = HyperParameterConfig(
            itr=i,
            n_confounder_rep=n_confounder,
            n_outcome_rep=n_cause + 1,
            mmd_sigma=1.0,
            lam_factual=1.0,
            lam_propensity=1.0,
            lam_mmd=1.0,
            batch_size=100,
            learning_rate=0.005,
        )
        param_list.append(config)

    return param_list


def get_scp_config_three(n_itr, n_confounder, p_confounder_cause):
    param_list = list()

    for i in range(n_itr):
        config = HyperParameterConfig(
            itr=i,
            n_confounder_rep=n_confounder // 4,
            n_outcome_rep=n_confounder // 2,
            mmd_sigma=1.0,
            lam_factual=1.0,
            lam_propensity=1.0,
            lam_mmd=1.0,
            learning_rate=0.005,
            batch_size=100,
        )
        param_list.append(config)

    return param_list


def get_scp_config_one(n_itr, n_confounder, n_cause):
    param_list = list()

    for i in range(n_itr):
        config = HyperParameterConfig(
            itr=i,
            n_confounder_rep=(n_confounder + n_cause + 1) // 2,
            n_outcome_rep=(n_confounder + n_cause + 1) // 2,
            mmd_sigma=1.0,
            lam_factual=1.0,
            lam_propensity=1.0,
            lam_mmd=1.0,
            batch_size=100,
            learning_rate=0.005,
        )
        param_list.append(config)

    return param_list


def get_nn_config_one(n_itr, n_confounder, n_cause):
    param_list = list()

    for i in range(n_itr):
        config = HyperParameterConfig(
            itr=i,
            n_confounder_rep=(n_confounder + n_cause + 1) // 2,
            n_outcome_rep=(n_confounder + n_cause + 1) // 2,
            mmd_sigma=1.0,
            lam_factual=1.0,
            lam_propensity=1.0,
            lam_mmd=1.0,
            batch_size=100,
            learning_rate=0.01,
        )
        param_list.append(config)

    return param_list
