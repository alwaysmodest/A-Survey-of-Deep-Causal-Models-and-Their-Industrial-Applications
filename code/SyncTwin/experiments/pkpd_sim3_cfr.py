import argparse

import numpy as np
import numpy.random
import torch

import CFR
import SyncTwin

# from config import DEVICE
from util import io_utils, train_utils

parser = argparse.ArgumentParser("PKPD simulation")
parser.add_argument("--seed", type=str, default="100")
parser.add_argument("--model_id", type=str, default="")
parser.add_argument("--itr", type=str, default="20")
parser.add_argument("--itr_pretrain", type=str, default="5000")
parser.add_argument("--itr_fine_tune", type=str, default="20000")
parser.add_argument("--batch_size", type=str, default="100")
parser.add_argument("--pretrain_Y", type=str, choices=["False", "True"], default="False")
parser.add_argument("--reduced_fine_tune", type=str, choices=["False", "True"], default="True")
parser.add_argument("--linear_decoder", type=str, choices=["False", "True"], default="False")
parser.add_argument("--lam_dist", type=str, default="1")
parser.add_argument("--n_hidden", type=str, default="20")
parser.add_argument("--sim_id", type=str)
parser.add_argument("--regular", type=str, choices=["False", "True"], default="True")


args = parser.parse_args()
seed = int(args.seed)
pretrain_Y = args.pretrain_Y == "True"
itr = int(args.itr)
itr_pretrain = int(args.itr_pretrain)
itr_fine_tune = int(args.itr_fine_tune)
batch_size = int(args.batch_size)
sim_id = args.sim_id
reduced_fine_tune = args.reduced_fine_tune == "True"
linear_decoder = args.linear_decoder == "True"
model_id = args.model_id
lam_dist = float(args.lam_dist)
n_hidden = int(args.n_hidden)
regular = args.regular == "True"
if not regular:
    n_hidden = n_hidden * 2

print("Running simulation with seed {}".format(seed))
numpy.random.seed(seed)
torch.manual_seed(seed)

if lam_dist == 0:
    model_id = "tarnet"
else:
    model_id = "cfr"

base_path_model = "models/{}-seed-".format(sim_id) + str(seed) + model_id
base_path_plot = "plots/{}-seed-".format(sim_id) + str(seed) + model_id
base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
train_utils.create_paths(base_path_model, base_path_plot, base_path_data)
plot_path = base_path_plot + "/unit-{}-dim-{}-{}.png"
data_path = base_path_data + "/{}-{}.{}"

# loading config and data
print("loading data")
n_units, n_treated, n_units_total, step, train_step, control_sample, noise, n_basis, n_cluster = io_utils.load_config(
    data_path, "train"
)
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
) = io_utils.load_tensor(data_path, "train")
(
    x_full_val,
    t_full_val,
    mask_full_val,
    batch_ind_full_val,
    y_full_val,
    y_control_val,
    y_mask_full_val,
    _,
    _,
    _,
) = io_utils.load_tensor(data_path, "val")

control_error_list = []
training_time_list = []

model_path = base_path_model + "/itr-" + str(0) + "-{}.pth"

if regular:
    enc = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=n_hidden)
else:
    enc = SyncTwin.GRUDEncoder(input_dim=3, hidden_dim=n_hidden)

dec_Y = SyncTwin.LinearDecoder(hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step)

cfr = CFR.CFR(n_units, n_treated, lam_dist=lam_dist, encoder=enc, decoder_Y=dec_Y)

print("training")

train_utils.train_cfr(
    cfr,
    x_full,
    t_full,
    mask_full,
    y_full,
    y_mask_full,
    x_full_val,
    t_full_val,
    mask_full_val,
    y_full_val,
    y_mask_full_val,
    niters=itr_pretrain,
    model_path=model_path,
)

print("Testing")

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
) = io_utils.load_tensor(data_path, "test")


# evaluating best model
effect_est = cfr.get_treatment_effect(x_full, t_full, mask_full, y_full, y_mask_full)
mae_effect = torch.mean(torch.abs(treatment_effect - effect_est)).item()
mae_sd = torch.std(torch.abs(treatment_effect - effect_est)).item() / np.sqrt(n_treated)
print("Treatment effect MAE: ({}, {})".format(mae_effect, mae_sd))
