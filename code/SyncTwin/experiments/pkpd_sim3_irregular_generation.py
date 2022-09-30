import argparse
import pickle

import numpy as np
import numpy.random
import torch
from sklearn.decomposition import PCA

# from config import DEVICE
from util import io_utils, train_utils

parser = argparse.ArgumentParser("PKPD simulation")
parser.add_argument("--sim_id", type=str)
parser.add_argument("--seed", type=str)
parser.add_argument("--missing_pct", type=str)

args = parser.parse_args()

missing_pct = float(args.missing_pct)
seed = int(args.seed)
sim_id = args.sim_id

base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
data_path = base_path_data + "/{}-{}.{}"

sim_id_export = sim_id + "-miss-" + str(missing_pct)
base_path_data_export = "data/{}-seed-".format(sim_id_export) + str(seed)
train_utils.create_paths(base_path_data_export)
data_path_export = base_path_data_export + "/{}-{}.{}"

numpy.random.seed(seed)
torch.manual_seed(seed)

n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster = io_utils.load_config(
    data_path, "train"
)

for fold in ["train", "val", "test"]:
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

    # setting up missing values
    missing = torch.rand_like(mask_full)

    missing = (missing < missing_pct) * 1.0

    mask_full_bool = mask_full + (1 - missing) == 2
    mask_full = mask_full_bool * 1.0

    # get benchmark
    x_full_zero = x_full.clone().detach()
    x_full_zero[~mask_full_bool] = 0

    pca = PCA(n_components=2)
    mat_in = x_full_zero.cpu().numpy()
    mat_in = mat_in.transpose((1, 0, 2)).reshape(x_full_zero.shape[1], x_full_zero.shape[0] * x_full_zero.shape[2])
    pca.fit(mat_in)
    mat_imputed = pca.inverse_transform(pca.transform(mat_in))
    mat_imputed = mat_imputed.reshape(x_full_zero.shape[1], x_full_zero.shape[0], x_full_zero.shape[2]).transpose(
        (1, 0, 2)
    )
    mat_imputed = torch.tensor(mat_imputed).to(mask_full)

    x_full_benchmark_input = x_full_zero + (1 - mask_full) * mat_imputed

    X0 = x_full_benchmark_input[:, :n_units, :]
    X0 = X0.permute((0, 2, 1)).reshape(X0.shape[0] * X0.shape[2], X0.shape[1]).cpu().numpy()

    X1 = x_full_benchmark_input[:, n_units:, :]
    X1 = X1.permute((0, 2, 1)).reshape(X1.shape[0] * X1.shape[2], X1.shape[1]).cpu().numpy()

    Y_control = y_control[:, :, 0].cpu().numpy()
    Y_treated = y_full[:, n_units:, 0].cpu().numpy()
    Treatment_effect = treatment_effect[:, :, 0].cpu().numpy()

    np.savetxt(data_path_export.format(fold, "X0", "csv"), X0, delimiter=",")
    np.savetxt(data_path_export.format(fold, "X1", "csv"), X1, delimiter=",")
    np.savetxt(data_path_export.format(fold, "Y_control", "csv"), Y_control, delimiter=",")
    np.savetxt(data_path_export.format(fold, "Y_treated", "csv"), Y_treated, delimiter=",")
    np.savetxt(data_path_export.format(fold, "Treatment_effect", "csv"), Treatment_effect, delimiter=",")
    torch.save(x_full_benchmark_input, data_path_export.format(fold, "x_full", "pth"))
    torch.save(t_full, data_path_export.format(fold, "t_full", "pth"))
    torch.save(mask_full, data_path_export.format(fold, "mask_full", "pth"))
    torch.save(batch_ind_full, data_path_export.format(fold, "batch_ind_full", "pth"))
    torch.save(y_full, data_path_export.format(fold, "y_full", "pth"))
    y_full_all = torch.load(data_path.format(fold, "y_full_all", "pth"))
    torch.save(y_full_all, data_path_export.format(fold, "y_full_all", "pth"))
    torch.save(y_control, data_path_export.format(fold, "y_control", "pth"))
    torch.save(y_mask_full, data_path_export.format(fold, "y_mask_full", "pth"))
    torch.save(m, data_path_export.format(fold, "m", "pth"))
    torch.save(sd, data_path_export.format(fold, "sd", "pth"))
    torch.save(treatment_effect, data_path_export.format(fold, "treatment_effect", "pth"))

    config = {
        "n_units": n_units,
        "n_treated": n_treated,
        "n_units_total": n_units_total,
        "step": step,
        "train_step": train_step,
        "control_sample": control_sample,
        "noise": noise,
        "n_basis": n_basis,
        "n_cluster": n_cluster,
    }
    with open(data_path_export.format(fold, "config", "pkl"), "wb") as f:
        pickle.dump(config, file=f)
