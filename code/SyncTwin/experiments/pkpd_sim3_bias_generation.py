import argparse
import pickle

import numpy as np
import numpy.random
import torch

from config import DEVICE
from sim import pkpd
from util import train_utils

parser = argparse.ArgumentParser("PKPD simulation: data generation")
parser.add_argument("--seed", type=str, default="100")
# sync3
parser.add_argument("--sim_id", type=str)
parser.add_argument("--train_step", type=str, default="25")
parser.add_argument("--step", type=str, default="30")
parser.add_argument("--control_sample", type=str, default="200")
parser.add_argument("--control_c1", type=str, default="100")
parser.add_argument("--treatment_sample", type=str, default="200")
parser.add_argument("--save_data", type=str, choices=["False", "True"], default="True")
parser.add_argument("--hidden_confounder", type=str, choices=["0", "1", "2", "3"], default="0")

args = parser.parse_args()
seed = int(args.seed)
save_data = args.save_data == "True"
sim_id = args.sim_id
train_step = int(args.train_step)
step = int(args.step)
control_sample = int(args.control_sample)
control_c1 = int(args.control_c1)
treatment_sample = int(args.treatment_sample)
control_c2 = control_sample - control_c1
hidden_confounder = int(args.hidden_confounder)

assert treatment_sample <= control_sample

print("Data generation with seed {}".format(seed))
numpy.random.seed(seed)
torch.manual_seed(seed)

# generate data
print("Generating data")
noise = 0.1
n_basis = 6
n_cluster = 2
base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
train_utils.create_paths(base_path_data)
data_path = base_path_data + "/{}-{}.{}"

for fold in ["test", "val", "train"]:
    Kin_list, Kin_b = pkpd.get_Kin(step=step, n_basis=n_basis)
    control_Kin_list, control_Kin_b = pkpd.get_clustered_Kin(
        Kin_b, n_cluster=n_cluster, n_sample_total=control_sample * 2
    )
    treat_Kin_list, treat_Kin_b = pkpd.get_clustered_Kin(Kin_b, n_cluster=n_cluster, n_sample_total=control_sample * 2)
    treat_Kin_list = treat_Kin_list[:treatment_sample]
    treat_Kin_b = treat_Kin_b[:, :treatment_sample]

    # build biased control samples
    control_Kin_list_c1 = control_Kin_list[:control_c1]
    control_Kin_b_c1 = control_Kin_b[:, :control_c1]
    control_Kin_list_c2 = control_Kin_list[control_sample : (control_sample + control_c2)]
    control_Kin_b_c2 = control_Kin_b[:, control_sample : (control_sample + control_c2)]
    print("Control C1: ", len(control_Kin_list_c1))
    print("Control C2: ", len(control_Kin_list_c2))

    control_Kin_list = control_Kin_list_c1 + control_Kin_list_c2
    control_Kin_b = np.concatenate([control_Kin_b_c1, control_Kin_b_c2], axis=1)
    print("control_Kin_b:", control_Kin_b.shape)

    K_list = [0.18]
    P0_list = [0.5]
    R0_list = [0.5]

    control_res_arr = pkpd.generate_data(
        control_Kin_list, K_list, P0_list, R0_list, train_step=-1, H=0.1, D50=0.1, step=step
    )
    treat_res_arr = pkpd.generate_data(
        treat_Kin_list, K_list, P0_list, R0_list, train_step=train_step, H=0.1, D50=0.1, step=step
    )
    treat_counterfactual_arr = pkpd.generate_data(
        treat_Kin_list, K_list, P0_list, R0_list, train_step=-1, H=0.1, D50=0.1, step=step
    )
    print("control_res_arr:", control_res_arr.shape)

    (
        n_tuple,
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        y_full_all,
        m,
        sd,
    ) = pkpd.get_covariate(
        control_Kin_b,
        treat_Kin_b,
        control_res_arr,
        treat_res_arr,
        step=step,
        train_step=train_step,
        device=DEVICE,
        noise=noise,
        double_up=False,
        hidden_confounder=hidden_confounder,
    )

    treatment_effect = pkpd.get_treatment_effect(treat_res_arr, treat_counterfactual_arr, train_step, m, sd)

    n_units, n_treated, n_units_total = n_tuple
    print(n_tuple)

    # export data to csv
    if save_data:

        X0 = x_full[:, :n_units, :]
        X0 = X0.permute((0, 2, 1)).reshape(X0.shape[0] * X0.shape[2], X0.shape[1]).cpu().numpy()

        X1 = x_full[:, n_units:, :]
        X1 = X1.permute((0, 2, 1)).reshape(X1.shape[0] * X1.shape[2], X1.shape[1]).cpu().numpy()

        Y_control = y_control[:, :, 0].cpu().numpy()
        Y_treated = y_full[:, n_units:, 0].cpu().numpy()
        Treatment_effect = treatment_effect[:, :, 0].cpu().numpy()

        np.savetxt(data_path.format(fold, "X0", "csv"), X0, delimiter=",")
        np.savetxt(data_path.format(fold, "X1", "csv"), X1, delimiter=",")
        np.savetxt(data_path.format(fold, "Y_control", "csv"), Y_control, delimiter=",")
        np.savetxt(data_path.format(fold, "Y_treated", "csv"), Y_treated, delimiter=",")
        np.savetxt(data_path.format(fold, "Treatment_effect", "csv"), Treatment_effect, delimiter=",")

        torch.save(x_full, data_path.format(fold, "x_full", "pth"))
        torch.save(t_full, data_path.format(fold, "t_full", "pth"))
        torch.save(mask_full, data_path.format(fold, "mask_full", "pth"))
        torch.save(batch_ind_full, data_path.format(fold, "batch_ind_full", "pth"))
        torch.save(y_full, data_path.format(fold, "y_full", "pth"))
        torch.save(y_full_all, data_path.format(fold, "y_full_all", "pth"))
        torch.save(y_control, data_path.format(fold, "y_control", "pth"))
        torch.save(y_mask_full, data_path.format(fold, "y_mask_full", "pth"))
        torch.save(m, data_path.format(fold, "m", "pth"))
        torch.save(sd, data_path.format(fold, "sd", "pth"))
        torch.save(treatment_effect, data_path.format(fold, "treatment_effect", "pth"))

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
        with open(data_path.format(fold, "config", "pkl"), "wb") as f:
            pickle.dump(config, file=f)
