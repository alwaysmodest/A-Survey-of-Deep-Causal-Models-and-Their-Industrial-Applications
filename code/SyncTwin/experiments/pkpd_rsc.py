import argparse

import numpy as np
import pandas as pd
import torch

from robust_synth import Synth
from util import io_utils


def get_rsc_data(data_path, fold, return_y=False):
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
    mat_x = torch.cat([x_full[:, :, 2:3], y_full], dim=0).squeeze()
    if not return_y:
        return mat_x
    else:
        return mat_x, y_full, treatment_effect


def get_val_hat(mat_x, n_units, train_step, n):
    mat_control = mat_x[:, :n_units]
    mat_input = mat_control.T
    df = pd.DataFrame(data=mat_input.cpu().numpy())

    y_est_list = []
    beta_list = []
    for i in range(n_units):
        lasso = Synth(i, year=train_step, num_sv=n, method="Lasso")
        lasso.fit(df)
        y0_est = lasso.mean
        beta = lasso.beta
        y_est_list.append(y0_est)
        beta_list.append(beta)
    y_est_mat = np.stack(y_est_list, axis=-1)[train_step:, :]
    return np.mean(np.abs(mat_control[train_step:, :].cpu().numpy() - y_est_mat))


def get_y_hat(mat_x, n_units, train_step, n):
    mat_control = mat_x[:, :n_units]

    y_est_list = []
    beta_list = []
    for i in range(n_treated):
        mat_treat = mat_x[:, (n_units + i) : (n_units + i + 1)]
        mat_input = torch.cat([mat_treat, mat_control], dim=1).T
        df = pd.DataFrame(data=mat_input.cpu().numpy())
        lasso = Synth(0, year=train_step, num_sv=n, method="Lasso")
        lasso.fit(df)
        y0_est = lasso.mean
        beta = lasso.beta
        y_est_list.append(y0_est)
        beta_list.append(beta)
    y_est_mat = np.stack(y_est_list, axis=-1)[train_step:, :]
    beta_mat = np.stack(beta_list, axis=-1)
    return y_est_mat, beta_mat


parser = argparse.ArgumentParser("RSC simulation")
parser.add_argument("--seed", type=str, default="100")
parser.add_argument("--sim_id", type=str)

args = parser.parse_args()
seed = int(args.seed)
sim_id = args.sim_id

base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
data_path = base_path_data + "/{}-{}.{}"
weight_path = base_path_data + "/{}.{}"


# loading config and data
print("loading data")
n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster = io_utils.load_config(
    data_path, "test"
)

mat_x_val = get_rsc_data(data_path, "val")
mat_x_test, y_full, treatment_effect = get_rsc_data(data_path, "test", True)


err_list = []
n_list = list(range(1, 5))

for n in n_list:
    err = get_val_hat(mat_x_val, n_units, train_step, n)
    err_list.append(err)

err = np.array(err_list)
best_ind = np.argmin(err)
best_n = n_list[best_ind]

y_est_mat, beta_mat = get_y_hat(mat_x_test, n_units, train_step, best_n)
te_est = y_full.squeeze().cpu().numpy()[:, n_units:] - y_est_mat
mae = np.mean(np.abs(treatment_effect.squeeze().cpu().numpy() - te_est))
mae_sd = np.std(np.abs(treatment_effect.squeeze().cpu().numpy() - te_est)) / np.sqrt(n_treated)

print("MAE {:.3f} ({:.3f})".format(mae, mae_sd))

np.savetxt(weight_path.format("w-rsc", ".csv"), beta_mat)
