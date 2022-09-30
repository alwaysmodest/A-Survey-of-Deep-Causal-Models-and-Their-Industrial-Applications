import argparse
import time

import numpy as np
import numpy.random
import torch

import SyncTwin
from config import DEVICE
from util import io_utils, train_utils
from util.batching_utils import get_splits

parser = argparse.ArgumentParser("CPRD experiment")
parser.add_argument("--seed", type=str, default="100")
parser.add_argument("--model_id", type=str, default="")
parser.add_argument("--itr", type=str, default="1")
parser.add_argument("--itr_pretrain", type=str, default="10000")
parser.add_argument("--itr_fine_tune", type=str, default="20000")
parser.add_argument("--batch_size", type=str, default="100")
parser.add_argument("--lam_prognostic", type=str, default="1")
parser.add_argument("--tau", type=str, default="1")
parser.add_argument("--n_hidden", type=str, default="50")
parser.add_argument("--fold", type=str, default="3")
parser.add_argument("--data_version", type=str, default="1")


args = parser.parse_args()
seed = int(args.seed)
# pretrain_Y = (args.pretrain_Y == 'True')
itr = int(args.itr)
itr_pretrain = int(args.itr_pretrain)
itr_fine_tune = int(args.itr_fine_tune)
batch_size = int(args.batch_size)
# sim_id = args.sim_id
# reduced_fine_tune = (args.reduced_fine_tune == 'True')
# linear_decoder = (args.linear_decoder == 'True')
model_id = args.model_id
lam_prognostic = float(args.lam_prognostic)
tau = float(args.tau)
n_hidden = int(args.n_hidden)
fold = int(args.fold)
data_version = int(args.data_version)

print("Running with seed {}".format(seed))
numpy.random.seed(seed)
torch.manual_seed(seed)


base_path_model = "models/real-data-seed-" + str(seed) + model_id
train_utils.create_paths(base_path_model)

# loading data

data1, data0 = io_utils.load_data_dict(data_version)

d1_train = get_splits(0, fold, list(data1.values()))
d1_val = get_splits(1, fold, list(data1.values()))
d1_test = get_splits(2, fold, list(data1.values()))

d0_train = get_splits(0, fold, list(data0.values()))
d0_val = get_splits(1, fold, list(data0.values()))
d0_test = get_splits(2, fold, list(data0.values()))

input_dim = d1_train[0].shape[-1]
train_step = d1_train[0].shape[1]
step = train_step + d1_train[-1].shape[1]


n_units_train, n_treated_train, n_units_total_train = io_utils.get_units(d1_train, d0_train)
print("Training: ", n_units_train, n_treated_train, n_units_total_train)

n_units_val, n_treated_val, n_units_total_val = io_utils.get_units(d1_val, d0_val)
print("Validation: ", n_units_val, n_treated_val, n_units_total_val)

n_units_test, n_treated_test, n_units_total_test = io_utils.get_units(d1_test, d0_test)
print("Testing: ", n_units_test, n_treated_test, n_units_total_test)


x_full, t_full, mask_full, batch_ind_full, y_full, y_control, y_mask_full, patid_full = io_utils.get_tensors(
    d1_train, d0_train, DEVICE
)

(
    x_full_val,
    t_full_val,
    mask_full_val,
    batch_ind_full_val,
    y_full_val,
    y_control_val,
    y_mask_full_val,
    patid_full_val,
) = io_utils.get_tensors(d1_val, d0_val, DEVICE)


control_error_list = []
training_time_list = []


for i in range(itr):
    print("Iteration {}".format(i))
    start_time = time.time()

    model_path = base_path_model + "/itr-" + str(i) + "-{}.pth"

    enc = SyncTwin.GRUDEncoder(input_dim=input_dim, hidden_dim=n_hidden)
    dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)

    dec_Y = SyncTwin.LinearDecoder(
        hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
    )

    nsc = SyncTwin.SyncTwin(
        n_units_train,
        n_treated_train,
        reg_B=0.0,
        lam_express=1.0,
        lam_recon=1.0,
        lam_prognostic=lam_prognostic,
        tau=tau,
        encoder=enc,
        decoder=dec,
        decoder_Y=dec_Y,
        reduce_gpu_memory=True,
    )

    final_loss = train_utils.pre_train_reconstruction_prognostic_loss(
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

    end_time = time.time()
    training_time = end_time - start_time
    training_time_list.append(training_time)
    print("--- Training done in %s seconds ---" % training_time)

    print("Pre-training loss: {}".format(final_loss))
    control_error_list.append(final_loss)


# find the best model and summarize
control_error = np.array(control_error_list)
best_itr = np.argmin(control_error)

model_path = base_path_model + "/itr-" + str(best_itr) + "-{}.pth"
enc = SyncTwin.GRUDEncoder(input_dim=input_dim, hidden_dim=n_hidden)
dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)
dec_Y = SyncTwin.LinearDecoder(hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step)

enc.load_state_dict(torch.load(model_path.format("encoder.pth")))
dec.load_state_dict(torch.load(model_path.format("decoder.pth")))
dec_Y.load_state_dict(torch.load(model_path.format("decoder_Y.pth")))

best_model_path = base_path_model + "/best-{}.pth"
torch.save(enc.state_dict(), best_model_path.format("encoder.pth"))
torch.save(dec.state_dict(), best_model_path.format("decoder.pth"))
torch.save(dec_Y.state_dict(), best_model_path.format("decoder_Y.pth"))

print("Best error: ", np.min(control_error))
print("Best itr: ", best_itr)
