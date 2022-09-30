import argparse
import sys

import numpy as np
import pandas as pds
import tensorflow as tf

sys.path.append("clairvoyance")
from datasets import dataset  # type: ignore  # noqa: E402
from preprocessing import ProblemMaker  # type: ignore  # noqa: E402
from treatments.treatments import treatment_effects_model  # type: ignore  # noqa: E402

from clair_helper import get_clair_data, silence_tf  # noqa: E402

silence_tf()

parser = argparse.ArgumentParser("Clair Benchmark")
parser.add_argument("--seed", type=str, default="100")
parser.add_argument("--batch_size", type=str, default="100")
parser.add_argument("--sim_id", type=str)
parser.add_argument("--model_name", type=str, default="CRN", choices=["CRN", "RMSN", "GARNITE"])
parser.add_argument("--max_alpha", type=str, default="1.0")
parser.add_argument("--n_hidden", type=str, default="128")


args = parser.parse_args()
seed = int(args.seed)
batch_size = int(args.batch_size)
sim_id = args.sim_id
model_name = args.model_name
max_alpha = float(args.max_alpha)
n_hidden = int(args.n_hidden)

df, df_static, max_seq_len, projection_horizon, n_units, n_units_total, treatment_effect = get_clair_data(
    seed, sim_id, "test"
)

tf.set_random_seed(seed)

df_train, df_static_train, _, _, _, _, _ = get_clair_data(seed, sim_id, "train")
df_val, df_static_val, _, _, _, _, _ = get_clair_data(seed, sim_id, "val")
df_static_val["id"] = df_static_val["id"] + 1e6
df_val["id"] = df_val["id"] + 1e6
df_static_train = pds.concat([df_static_train, df_static_val], ignore_index=True)
df_train = pds.concat([df_train, df_val], ignore_index=True)

dataset_training = dataset.PandasDataset(static_data=df_static_train, temporal_data=df_train)
dataset_testing = dataset.PandasDataset(static_data=df_static, temporal_data=df)


# Define parameters
problem = "online"
label_name = "outcome"
treatment = ["treatment"]
window = 1

# Define problem
problem_maker = ProblemMaker(
    problem=problem, label=[label_name], max_seq_len=max_seq_len, treatment=treatment, window=window
)

dataset_training = problem_maker.fit_transform(dataset_training)
dataset_testing = problem_maker.fit_transform(dataset_testing)

# Set other parameters
metric_name = "mse"
task = "regression"

metric_sets = [metric_name]
metric_parameters = {"problem": problem, "label_name": [label_name]}

print("Finish defining problem.")

# Set up validation for early stopping and best model saving
dataset_training.train_val_test_split(prob_val=0.5, prob_test=0.0)

if model_name == "CRN":
    hyperparams_encoder = {
        "rnn_hidden_units": n_hidden,
        "br_size": 64,
        "fc_hidden_units": n_hidden,
        "learning_rate": 0.001,
        "batch_size": batch_size,
        "rnn_keep_prob": 0.9,
        "num_epochs": 300,
        "max_alpha": max_alpha,
    }

    hyperparams_decoder = {
        "br_size": 64,
        "fc_hidden_units": n_hidden,
        "learning_rate": 0.001,
        "batch_size": batch_size,
        "rnn_keep_prob": 0.9,
        "num_epochs": 300,
        "max_alpha": max_alpha,
    }

    model_parameters = {
        "hyperparams_encoder": hyperparams_encoder,
        "hyperparams_decoder": hyperparams_decoder,
        "static_mode": "concatenate",
        "time_mode": "concatenate",
    }
    treatment_model = treatment_effects_model(model_name, model_parameters, task=task)
    treatment_model.fit(dataset_training, projection_horizon=projection_horizon)

elif model_name == "RMSN":
    hyperparams_encoder_iptw = {
        "dropout_rate": 0.1,
        "memory_multiplier": 4,
        "num_epochs": 100,
        "batch_size": batch_size,
        "learning_rate": 0.01,
        "max_norm": 0.5,
    }

    hyperparams_decoder_iptw = {
        "dropout_rate": 0.1,
        "memory_multiplier": 2,
        "num_epochs": 100,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "max_norm": 4.0,
    }

    model_parameters = {
        "hyperparams_encoder_iptw": hyperparams_encoder_iptw,
        "hyperparams_decoder_iptw": hyperparams_decoder_iptw,
        "static_mode": "concatenate",
        "time_mode": "concatenate",
        "model_dir": "tmp/",
        "model_name": "rmsn_test",
    }
    treatment_model = treatment_effects_model(model_name, model_parameters, task=task)
    treatment_model.fit(dataset_training, projection_horizon=projection_horizon)

treatment_options = np.zeros((2, projection_horizon + 1, 1))
treatment_options[1, :, 0] = 1.0

y0_list = []

for patid in range(n_units, n_units_total):
    if model_name in ["CRN", "RMSN"]:
        # Predict and visualize counterfactuals for the sequence of treatments indicated by the user through the treatment_options
        history, counterfactual_traj = treatment_model.predict_counterfactual_trajectories(
            dataset=dataset_testing,
            patient_id=patid,
            timestep=max_seq_len - projection_horizon,
            treatment_options=treatment_options,
            test_split="train",
        )
        y0 = counterfactual_traj[1, :projection_horizon, 0] - counterfactual_traj[0, :projection_horizon, 0]
        y0_list.append(y0)

te_est = np.stack(y0_list, axis=-1)
mae = np.mean(np.abs(te_est - treatment_effect.squeeze().cpu().numpy()))
mae_sd = np.std(np.abs(te_est - treatment_effect.squeeze().cpu().numpy())) / np.sqrt(n_units_total - n_units)
print("Treatment effect MAE: {} SE: {}".format(mae, mae_sd))
