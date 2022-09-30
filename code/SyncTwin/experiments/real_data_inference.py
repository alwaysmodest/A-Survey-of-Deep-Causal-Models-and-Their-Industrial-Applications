import argparse

import numpy as np
import pandas as pds
import torch

import SyncTwin
from config import DEVICE
from util import eval_utils, io_utils, train_utils
from util.batching_utils import get_split_inference, get_splits

parser = argparse.ArgumentParser("Real data inference")
parser.add_argument("--seed", type=str, default="100")
parser.add_argument("--model_id", type=str, default="m100-v3")
parser.add_argument("--itr_fine_tune", type=str, default="100")
parser.add_argument("--batch_size", type=str, default="1000")
parser.add_argument("--tau", type=str, default="1")
parser.add_argument("--n_hidden", type=str, default="100")
parser.add_argument("--data_version", type=str, default="3")
parser.add_argument("--lr", type=str, default="0.01")

args = parser.parse_args()
seed = int(args.seed)
itr_fine_tune = int(args.itr_fine_tune)
batch_size = int(args.batch_size)
model_id = args.model_id
tau = float(args.tau)
n_hidden = int(args.n_hidden)
version = int(args.data_version)
lr = float(args.lr)

lam_prognostic = 1.0
fold = 3


base_path_model = "models/real-data-seed-" + str(seed) + model_id

# loading data

data1, data0 = io_utils.load_data_dict(version)

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
# get a subset of validation data

n0_val = n_units_val
n1_val = batch_size

patid_full_val = torch.tensor(patid_full_val)
val_full = [
    x_full_val,
    t_full_val,
    mask_full_val,
    batch_ind_full_val,
    y_full_val,
    y_control_val,
    y_mask_full_val,
    patid_full_val,
]


res = get_split_inference(n0_val, n1_val, n_units_val, val_full, 0)

# this is to validation on a separate control group
# res2 = get_split(n0_val, n1_val, n0_val, val_full, 1)

ite_list = []
patid_list = []

# this is to get treated group
n_max = n_treated_val // n1_val
print("Batch size: ", n1_val)
print("Number of Batches: ", n_max)

for n in range(n_max + 1):
    print("Batch: ", str(n))
    res2 = get_split_inference(n0_val, n1_val, n_units_val, val_full, 1, n)

    (  # pylint: disable=unbalanced-tuple-unpacking
        x_full_val0,
        t_full_val0,
        mask_full_val0,
        batch_ind_full_val0,
        y_full_val0,
        y_control_val0,
        y_mask_full_val0,
        patid_full_val0,
    ) = res
    (  # pylint: disable=unbalanced-tuple-unpacking
        x_full_val1,
        t_full_val1,
        mask_full_val1,
        batch_ind_full_val1,
        y_full_val1,
        y_control_val1,
        y_mask_full_val1,
        patid_full_val1,
    ) = res2
    print("Shape x_full_val1: ", x_full_val1.shape)
    actual_size = x_full_val1.shape[1]

    enc = SyncTwin.GRUDEncoder(input_dim=input_dim, hidden_dim=n_hidden)
    dec = SyncTwin.LSTMTimeDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=train_step)

    dec_Y = SyncTwin.LinearDecoder(
        hidden_dim=enc.hidden_dim, output_dim=y_full.shape[-1], max_seq_len=step - train_step
    )

    nsc = SyncTwin.SyncTwin(
        n0_val,
        actual_size,
        reg_B=0.0,
        lam_express=1.0,
        lam_recon=1.0,
        lam_prognostic=lam_prognostic,
        tau=tau,
        encoder=enc,
        decoder=dec,
        decoder_Y=dec_Y,
        inference_only=True,
    )

    model_path = base_path_model + "/best-{}.pth"
    model_path_save = base_path_model + "/inference-itr-" + str(n) + "-{}.pth"
    print("loading model from: ", model_path)
    print("saving model at: ", model_path_save)

    train_utils.load_pre_train_and_init(
        nsc,
        x_full_val0,
        t_full_val0,
        mask_full_val0,
        batch_ind_full_val0.to(torch.long),
        model_path=model_path,
        init_decoder_Y=True,
    )
    return_code = train_utils.train_B_self_expressive(
        nsc,
        x_full_val1,
        t_full_val1,
        mask_full_val1,
        torch.arange(actual_size).to(torch.long),
        niters=itr_fine_tune,
        model_path=model_path_save,
        lr=lr,
    )

    with torch.no_grad():
        y_hat = eval_utils.get_prediction(nsc, torch.arange(actual_size).to(torch.long), y_control_val0, itr=500).cpu()

    ite = (y_full_val1 - y_hat).cpu().numpy().squeeze()
    patid = patid_full_val1.cpu().numpy()

    ite_list.append(ite)
    patid_list.append(patid)

patid = np.concatenate(patid_list, axis=0)
ite = np.concatenate(ite_list, axis=0)
att = np.mean(ite)
print("ATT: ", att)
print("ITE SD: ", np.std(ite))

df_ite = pds.DataFrame(data=np.stack((patid, ite), axis=1), columns=["patid", "ite"])
df_ite.head()
df_ite.to_csv("real_data/model-{}-version-{}-ite-est.csv".format(model_id, version))
print("Estimated ATE:", df_ite.ite.mean())
