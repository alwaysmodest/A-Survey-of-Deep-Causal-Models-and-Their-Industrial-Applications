import argparse

import GPy
import numpy as np
import torch

from util import io_utils

parser = argparse.ArgumentParser("CGP simulation")
parser.add_argument("--seed", type=str, default="100")
parser.add_argument("--sim_id", type=str)

args = parser.parse_args()
seed = int(args.seed)
sim_id = args.sim_id


def load_feat_gp(data_path, fold):
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
    x_feat = x_full.permute((1, 0, 2)).reshape(x_full.shape[1], x_full.shape[0] * x_full.shape[2]).cpu().numpy()

    treat_ind = np.zeros(n_units_total)
    treat_ind[n_units:] = 1

    counter_ind = 1 - treat_ind
    x_feat = np.concatenate([x_feat, treat_ind[:, None]], axis=-1)
    x_counter = np.concatenate([x_feat, counter_ind[:, None]], axis=-1)
    return y_full, x_feat, x_counter, treatment_effect


base_path_data = "data/{}-seed-".format(sim_id) + str(seed)

data = "data/{}-seed-".format(sim_id) + str(seed)
data_path = base_path_data + "/{}-{}.{}"
weight_path = base_path_data + "/{}.{}"


# loading config and data
print("loading data")
n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster = io_utils.load_config(
    data_path, "test"
)
y_full, x_feat, x_counter, _ = load_feat_gp(data_path, "train")
y_full_val, x_feat_val, x_counter_val, _ = load_feat_gp(data_path, "val")
y_full_test, x_feat_test, x_counter_test, treatment_effect = load_feat_gp(data_path, "test")
y_full = torch.cat((y_full, y_full_val), dim=1)
x_feat = np.concatenate((x_feat, x_feat_val), axis=0)

te_est_list = []
for time_step in range(5):
    Y_train = y_full[time_step, :, 0].cpu().numpy()[:, None]
    Y_test = y_full_test[time_step, :, 0].cpu().numpy()[:, None]
    kernel = GPy.kern.RBF(input_dim=x_feat.shape[1], variance=np.var(Y_train), lengthscale=1.0)
    gp_model = GPy.models.GPRegression(x_feat, Y_train, kernel=kernel, noise_var=np.var(Y_train))
    gp_model.optimize(messages=False)
    gp_pred_d, gp_pred_d_var = gp_model.predict(Xnew=x_counter_test)

    te_est = (Y_test - gp_pred_d)[n_units:, ...]
    te_est_list.append(te_est)

te_est = np.concatenate(te_est_list, axis=-1)
mae = np.mean(np.abs(te_est.T - treatment_effect.cpu().numpy().squeeze()))
sd = np.std(np.abs(te_est.T - treatment_effect.cpu().numpy().squeeze())) / np.sqrt(n_treated)

print("MAE: {:.3f} ({:.3f})".format(mae, sd))
