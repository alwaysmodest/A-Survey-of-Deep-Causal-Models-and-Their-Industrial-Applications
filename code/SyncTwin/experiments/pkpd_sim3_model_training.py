import argparse
import time

import numpy as np
import numpy.random
import torch

import SyncTwin

# from config import DEVICE
from util import eval_utils, io_utils, train_utils

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
parser.add_argument("--lam_prognostic", type=str, default="1")
parser.add_argument("--lam_recon", type=str, default="1")
parser.add_argument("--tau", type=str, default="1")
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
lam_prognostic = float(args.lam_prognostic)
lam_recon = float(args.lam_recon)
tau = float(args.tau)
n_hidden = int(args.n_hidden)
regular = args.regular == "True"
if not regular:
    n_hidden = n_hidden * 2

print("Running simulation with seed {}".format(seed))
numpy.random.seed(seed)
torch.manual_seed(seed)

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


for i in range(itr):
    print("Iteration {}".format(i))
    start_time = time.time()

    model_path = base_path_model + "/itr-" + str(i) + "-{}.pth"

    if regular:
        enc = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=n_hidden)
        dec = SyncTwin.RegularDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
    else:
        enc = SyncTwin.GRUDEncoder(input_dim=3, hidden_dim=n_hidden)
        dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)

    if pretrain_Y:
        if not linear_decoder:
            dec_Y = SyncTwin.RegularDecoder(
                hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
            )
        else:
            dec_Y = SyncTwin.LinearDecoder(
                hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
            )
    else:
        dec_Y = None

    nsc = SyncTwin.SyncTwin(
        n_units,
        n_treated,
        reg_B=0.0,
        lam_express=1.0,
        lam_recon=lam_recon,
        lam_prognostic=lam_prognostic,
        tau=tau,
        encoder=enc,
        decoder=dec,
        decoder_Y=dec_Y,
    )

    print("Pretrain")
    if not pretrain_Y:
        train_utils.pre_train_reconstruction_loss(
            nsc,
            x_full,
            t_full,
            mask_full,
            x_full_val,
            t_full_val,
            mask_full_val,
            niters=itr_pretrain,
            model_path=model_path,
            batch_size=batch_size,
        )
    else:
        train_utils.pre_train_reconstruction_prognostic_loss(
            nsc,
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
            batch_size=batch_size,
        )

    if not reduced_fine_tune:
        return_code = train_utils.train_all_losses(
            nsc,
            x_full,
            t_full,
            mask_full,
            batch_ind_full,
            y_full,
            y_control,
            y_mask_full,
            niters=itr_pretrain,
            model_path=model_path,
            batch_size=None,
        )

    print("Eval")
    print("Matching on validation set")
    train_utils.load_pre_train_and_init(
        nsc, x_full_val, t_full_val, mask_full_val, batch_ind_full_val, model_path=model_path, init_decoder_Y=pretrain_Y
    )
    return_code = train_utils.train_B_self_expressive(
        nsc, x_full_val, t_full_val, mask_full_val, batch_ind_full_val, niters=itr_fine_tune, model_path=model_path
    )

    if return_code != 0:
        control_error_list.append(1e9)
        continue

    end_time = time.time()
    training_time = end_time - start_time
    training_time_list.append(training_time)
    print("--- Training done in %s seconds ---" % training_time)

    train_utils.load_nsc(nsc, x_full_val, t_full_val, mask_full_val, batch_ind_full_val, model_path=model_path)

    effect_est, y_hat = eval_utils.get_treatment_effect(nsc, batch_ind_full_val, y_full_val, y_control_val)
    y_control_val = y_control_val.to(y_hat.device)
    control_error = torch.mean(torch.abs(y_control_val - y_hat[:, :control_sample, :])).item()
    print("Control Y MAE: {}".format(control_error))
    control_error_list.append(control_error)

print("Testing")

# find the best model and summarize
control_error = np.array(control_error_list)
best_itr = np.argmin(control_error)

model_path = base_path_model + "/itr-" + str(best_itr) + "-{}.pth"
if regular:
    enc = SyncTwin.RegularEncoder(input_dim=3, hidden_dim=n_hidden)
    dec = SyncTwin.RegularDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
else:
    enc = SyncTwin.GRUDEncoder(input_dim=3, hidden_dim=n_hidden)
    dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
if pretrain_Y:
    if not linear_decoder:
        dec_Y = SyncTwin.RegularDecoder(
            hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
        )
    else:
        dec_Y = SyncTwin.LinearDecoder(
            hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
        )
else:
    dec_Y = None

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


control_error_list = []
for i in range(itr):
    print("Iteration {}".format(i))
    model_path_save = base_path_model + "/itr-" + str(i) + "-{}-testing.pth"

    nsc = SyncTwin.SyncTwin(
        n_units,
        n_treated,
        reg_B=0.0,
        lam_express=1.0,
        lam_recon=lam_recon,
        lam_prognostic=lam_prognostic,
        tau=tau,
        encoder=enc,
        decoder=dec,
        decoder_Y=dec_Y,
    )
    train_utils.load_pre_train_and_init(
        nsc, x_full, t_full, mask_full, batch_ind_full, model_path=model_path, init_decoder_Y=pretrain_Y
    )
    return_code = train_utils.train_B_self_expressive(
        nsc, x_full, t_full, mask_full, batch_ind_full, niters=itr_fine_tune, model_path=model_path_save
    )
    if return_code != 0:
        control_error_list.append(1e9)
        continue
    train_utils.load_nsc(nsc, x_full, t_full, mask_full, batch_ind_full, model_path=model_path_save)

    effect_est, y_hat = eval_utils.get_treatment_effect(nsc, batch_ind_full, y_full, y_control)
    control_error = torch.mean(torch.abs(y_control - y_hat[:, :control_sample, :])).item()
    control_error_list.append(control_error)

control_error = np.array(control_error_list)
best_itr = np.argmin(control_error)
model_path = base_path_model + "/itr-" + str(best_itr) + "-{}-testing.pth"
train_utils.load_nsc(nsc, x_full, t_full, mask_full, batch_ind_full, model_path=model_path)

best_model_path = base_path_model + "/best-{}.pth"
torch.save(nsc.state_dict(), best_model_path.format("nsc.pth"))

# evaluating best model
effect_est, y_hat = eval_utils.get_treatment_effect(nsc, batch_ind_full, y_full, y_control)
mae_effect = torch.mean(torch.abs(treatment_effect - effect_est)).item()
mae_sd = torch.std(torch.abs(treatment_effect - effect_est)).item() / np.sqrt(n_treated)
print("Best model: {}".format(str(best_itr)))
print("Treatment effect MAE: ({}, {})".format(mae_effect, mae_sd))
print("Control Y MAE: {}".format(np.min(control_error)))

# eval_utils.summary_simulation(nsc, None, x_full, t_full, mask_full, batch_ind_full, y_full, y_control, plot_path=plot_path)

mat_b = nsc.get_B_reduced(batch_ind_full).detach().cpu().numpy()
mat_b = mat_b[n_units:, :]

eval_utils.summarize_B(mat_b)

training_time = np.array(training_time_list)
avg_training_time = np.mean(training_time)
sd_training_time = np.std(training_time)
print("Average Training Time: {}".format(avg_training_time))
print("Std Training Time: {}".format(sd_training_time))
