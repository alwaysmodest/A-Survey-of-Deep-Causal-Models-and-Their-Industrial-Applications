import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn

import sim_config
from mlp import DirectOutcomeRegression, ModelTrainer
from sim_data_gen import DataGenerator
from utils import bootstrap_RMSE, get_scp_config, load_model


def run(d_config, eval_only=False, n_ensemble=None):
    n_confounder = d_config.n_confounder
    n_cause = d_config.n_cause
    n_outcome = d_config.n_outcome
    sample_size = d_config.sample_size
    p_confounder_cause = d_config.p_confounder_cause
    p_cause_cause = d_config.p_cause_cause
    cause_noise = d_config.cause_noise
    outcome_noise = d_config.outcome_noise
    linear = d_config.linear
    n_flip = d_config.n_flip
    p_outcome_single = d_config.p_outcome_single
    p_outcome_double = d_config.p_outcome_double
    outcome_interaction = d_config.outcome_interaction

    batch_size = 100
    max_epoch = 100
    model_id = "DOR-ens"
    model_path = "model/{}_{}_model/".format(model_id, d_config.sim_id)
    if not eval_only:
        try:
            shutil.rmtree(model_path)
        except OSError as e:
            print("shutil note: %s - %s." % (e.filename, e.strerror))

    learning_rate = 0.01
    seed = 100
    if n_ensemble is None:
        hyper_param_itr = n_cause
    else:
        hyper_param_itr = int(n_ensemble) + 1

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ratio = sample_size / 1000

    dg = DataGenerator(
        n_confounder,
        n_cause,
        n_outcome,
        sample_size,
        p_confounder_cause,
        p_cause_cause,
        cause_noise,
        outcome_noise,
        linear=linear,
        confounding_level=d_config.confounding_level,
        real_data=d_config.real_data,
        train_frac=0.7 / train_ratio,
        val_frac=0.1 / train_ratio,
        p_outcome_single=p_outcome_single,
        p_outcome_double=p_outcome_double,
        outcome_interaction=outcome_interaction,
    )

    train_dataset, valid_dataset, x_test, y_test = dg.generate_dataset()

    if not d_config.real_data:
        new_x_test, cate_test = dg.generate_counterfactual_test(n_flip)

    new_x_list = dg.generate_test_real()

    rmse = nn.MSELoss()

    param_list = get_scp_config(hyper_param_itr, n_confounder, p_confounder_cause)
    y_hat_list = []
    cate_hat_list = []
    for param in param_list:
        model_id_to_save = model_id + "_itr_{}".format(param.itr)

        # model = DirectOutcomeRegression(n_confounder, n_cause, n_outcome, n_hidden=param.n_outcome_rep + param.n_confounder_rep)
        model = DirectOutcomeRegression(
            n_confounder, n_cause, n_outcome, n_hidden=n_confounder + n_cause + 1, linear=False
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

        trainer = ModelTrainer(100, max_epoch, rmse, model_id_to_save, model_path)

        trainer.train(model, optimizer, train_dataset, valid_dataset, print_every=10)

        load_model(model, model_path, model_id_to_save)
        y_list = []
        with torch.no_grad():
            y_hat0 = model(x_test)
            y_hat1 = model(new_x_test)
            cate_hat = y_hat1 - y_hat0
            cate_hat_list.append(cate_hat)
            for i in range(len(new_x_list)):
                new_x = new_x_list[i]
                y_hat = model(new_x)
                y_list.append(y_hat)
            y_mat = torch.stack(y_list, dim=-1)
            y_hat_list.append(y_mat)

    cate_hat_mat = torch.stack(cate_hat_list, dim=-1)
    cate_hat = torch.mean(cate_hat_mat, dim=-1)
    y_hat_mat = torch.stack(y_hat_list, dim=-1)
    y_mat = torch.mean(y_hat_mat, dim=-1).cpu().numpy()
    y_mat = y_mat[:, 0, :]

    if not d_config.real_data:
        with torch.no_grad():

            error = torch.sqrt(rmse(cate_hat, cate_test))
            rmse_sd = bootstrap_RMSE((cate_hat - cate_test) ** 2)

            y_mat_true = np.concatenate(dg.outcome_list, axis=-1)
            print("y_mat_true", y_mat_true.shape)
            print("y_mat", y_mat.shape)
            # N, 2^K
            n_test = y_mat.shape[0]
            err_all = np.sum((y_mat_true[-n_test:, :] - y_mat) ** 2, axis=1)
            rmse_all = np.sqrt(np.mean(err_all))
            rmse_all_sd = bootstrap_RMSE(torch.tensor(err_all))  # pylint: disable=not-callable
            print(round(error.item(), 3), round(rmse_sd, 3), round(rmse_all, 3), round(rmse_all_sd, 3))
    else:
        y_list = []
        for i in range(y_mat.shape[1]):
            y_list.append(y_mat[:, i + i + 1])
        dg.evaluate_real(y_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ablation")
    parser.add_argument("--ablation", type=str, default="None")
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    i = args.ablation.find("perturb_subset")
    if i >= 0:
        subset = int(args.ablation.split("-")[-1])
        ablation = args.ablation[:i]

    config_key = args.config

    try:
        config1 = sim_config.sim_dict[config_key]
    except KeyError:
        print(config_key)
        exit(-1)

    run(config1, eval_only=False, n_ensemble=subset)

    # config1 = sim_config.sim_dict['n_confounder_10_linear']
    # config1 = sim_config.sim_dict['real_3000']
    # run(config1, eval_only=False)
