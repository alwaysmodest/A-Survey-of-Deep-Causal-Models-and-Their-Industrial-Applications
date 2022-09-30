import argparse

import pandas as pds

from util import eval_utils, io_utils

parser = argparse.ArgumentParser("Synth Benchmark")
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

dat_w = pds.read_csv(weight_path.format("w", "csv"), index_col=0)
dat_w2 = pds.read_csv(weight_path.format("w-nn", "csv"), index_col=0)
mat_w2 = dat_w2.values.transpose()
mat_w = dat_w.values.transpose()

print("Synthetic Control")
mae = eval_utils.effect_mae_from_w(mat_w, n_units, y_control, y_full, treatment_effect)
print("MAE: {}".format(mae))
eval_utils.summarize_B(mat_w, False)

print("1NN Matching")
mae = eval_utils.effect_mae_from_w(mat_w2, n_units, y_control, y_full, treatment_effect)
print("MAE: {}".format(mae))
eval_utils.summarize_B(mat_w2, False)
