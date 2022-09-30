import argparse

import numpy.random
import pandas as pds
import torch

import SyncTwin
from util import eval_utils, io_utils, train_utils

parser = argparse.ArgumentParser("PKPD simulation")
parser.add_argument("--sim_id", type=str)
args = parser.parse_args()
sim_id = args.sim_id

print(sim_id)
seed = 100
base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
data_path2 = base_path_data + "/{}.{}"
numpy.random.seed(seed)
torch.manual_seed(seed)


def get_summary(model_id):

    base_path_model = "models/{}-seed-".format(sim_id) + str(seed) + model_id
    base_path_plot = "plots/{}-seed-".format(sim_id) + str(seed) + model_id

    train_utils.create_paths(base_path_model, base_path_plot, base_path_data)
    data_path = base_path_data + "/{}-{}.{}"

    # loading config and data
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
    ) = io_utils.load_config(data_path)
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
    ) = io_utils.load_tensor(data_path)
    best_model_path = base_path_model + "/best-{}.pth"
    enc = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=20)
    dec = SyncTwin.RegularDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
    dec_Y = SyncTwin.LinearDecoder(
        hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
    )
    if model_id.find("none") > 0:
        nsc = SyncTwin.SyncTwin(
            n_units,
            n_treated,
            reg_B=0.0,
            lam_express=1.0,
            lam_recon=1.0,
            lam_prognostic=0.0,
            tau=0.3,
            encoder=enc,
            decoder=dec,
        )
    else:
        nsc = SyncTwin.SyncTwin(
            n_units,
            n_treated,
            reg_B=0.0,
            lam_express=1.0,
            lam_recon=1.0,
            lam_prognostic=0.0,
            tau=0.3,
            encoder=enc,
            decoder=dec,
            decoder_Y=dec_Y,
        )

    train_utils.load_nsc(nsc, x_full, t_full, mask_full, batch_ind_full, model_path=best_model_path.format("nsc.pth"))
    mat_b = nsc.get_B_reduced(batch_ind_full).detach().cpu().numpy()
    mat_b = mat_b[n_units:, :]
    return eval_utils.summarize_B(mat_b, show_plot=False)


print("Full model")
get_summary("-prognostic-linear")
print("\n", "Prognostic None")
get_summary("-prognostic-none")
print("\n", "Prognostic-recon")
get_summary("-prognostic-recon")
print("\n", "SC")
dat_w = pds.read_csv(data_path2.format("w", "csv"), index_col=0)
mat_w = dat_w.values.transpose()
eval_utils.summarize_B(mat_w, show_plot=False)
