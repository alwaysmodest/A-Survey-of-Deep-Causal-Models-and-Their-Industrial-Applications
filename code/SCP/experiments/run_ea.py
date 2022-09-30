import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn

import sim_config
import sim_data_gen
from mlp import DirectOutcomeRegression, ModelTrainer
from utils import bootstrap_RMSE, get_scp_config, load_model


def _add_noise_cause(k_cause, cause_mat, p):
    mask = np.random.binomial(1, p, cause_mat.shape)
    mask[:, :k_cause] = 0
    new_cause = cause_mat * (1 - mask) + (1.0 - cause_mat) * mask
    return new_cause


def add_noise_cause(nc_list, p):
    new_nc_list = []
    for i in range(len(nc_list)):
        new_cause = _add_noise_cause(i, nc_list[i], p)
        new_nc_list.append(new_cause)
    return new_nc_list


def reshape_arr(a):
    at = np.transpose(a, (0, 2, 1))
    return at.reshape(at.shape[0] * at.shape[1], at.shape[-1])


def add_noise_outcome(no_list, sigma):
    new_no_list = []
    for i in range(len(no_list)):
        o = no_list[i]
        noise = np.random.randn(o.shape[0], o.shape[1]) * sigma
        new_no_list.append(o + noise)
    return new_no_list


def run(d_config, p=0, sigma=0, eval_only=False, revert="n", full_balance=False):

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
    sample_size_train = d_config.sample_size_train
    seed = 100
    hyper_param_itr = 1
    param_list = get_scp_config(hyper_param_itr, n_confounder, p_confounder_cause)

    max_epoch = 500
    model_id = "EA-{:.2f}-{:.2f}".format(p, sigma)
    model_path = "model/{}_{}_model/".format(model_id, d_config.sim_id)
    if not eval_only:
        try:
            shutil.rmtree(model_path)
        except OSError as e:
            print("shutil note: %s - %s." % (e.filename, e.strerror))

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ratio = sample_size / 1000

    dg = sim_data_gen.DataGenerator(
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
    confounder, cause, outcome = dg.generate()

    nc_list = []
    no_list = []

    for k_cause in range(dg.n_cause):
        if full_balance:
            new_cause = _add_noise_cause(0, cause, 0.5)
        else:
            new_cause, _ = dg.get_x_potential_cause_oracle(k_cause, True)
        new_outcome = dg.generate_counterfactual(new_cause)
        nc_list.append(new_cause)
        no_list.append(new_outcome)

    nc_list_noise = add_noise_cause(nc_list, p)

    no_list_noise = add_noise_outcome(no_list, sigma)
    nc_list_noise.append(cause)
    no_list_noise.append(outcome)

    # B, D, N_CAUSE + 1
    if revert == "y":
        cause_cat = np.stack([cause] * (dg.n_cause + 1), axis=-1)
        outcome_cat = np.stack([outcome] * (dg.n_cause + 1), axis=-1)
        confounder_cat = np.stack([confounder] * (dg.n_cause + 1), axis=-1)
    else:
        cause_cat = np.stack(nc_list_noise, axis=-1)
        outcome_cat = np.stack(no_list_noise, axis=-1)
        confounder_cat = np.stack([confounder] * (dg.n_cause + 1), axis=-1)

    x_cat = np.concatenate([confounder_cat, cause_cat], axis=1)

    # get training and validation
    x_train = reshape_arr(x_cat[: dg.train_size])
    y_train = reshape_arr(outcome_cat[: dg.train_size])

    x_val = reshape_arr(x_cat[dg.train_size : dg.train_size + dg.val_size])
    y_val = reshape_arr(outcome_cat[dg.train_size : dg.train_size + dg.val_size])

    train_dataset = torch.utils.data.dataset.TensorDataset(
        dg._make_tensor(x_train), dg._make_tensor(y_train)  # pylint: disable=protected-access
    )
    valid_dataset = torch.utils.data.dataset.TensorDataset(
        dg._make_tensor(x_val), dg._make_tensor(y_val)  # pylint: disable=protected-access
    )

    dg.generate_counterfactual_test(n_flip)
    new_x_list = dg.generate_test_real()

    rmse = nn.MSELoss()

    err_list = list()
    for param in param_list:
        model_id_to_save = model_id + "_itr_{}".format(param.itr)

        model = DirectOutcomeRegression(
            n_confounder, n_cause, n_outcome, n_hidden=param.n_outcome_rep + param.n_confounder_rep
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=param.learning_rate)

        trainer = ModelTrainer(param.batch_size, max_epoch, rmse, model_id_to_save, model_path)

        trainer.train(model, optimizer, train_dataset, valid_dataset, print_every=10)

        load_model(model, model_path, model_id_to_save)

        with torch.no_grad():
            x_valid = valid_dataset.tensors[0]
            y_valid = valid_dataset.tensors[1]

            y_hat = model(x_valid)
            error = torch.sqrt(rmse(y_hat, y_valid))
            err_list.append(error.item())

    # select model with best hyper-parameter
    best_index = int(np.argmin(np.array(err_list)))
    best_param = param_list[best_index]
    print("Best param:", best_param)
    model_id_to_load = model_id + "_itr_{}".format(best_param.itr)
    model = DirectOutcomeRegression(
        n_confounder, n_cause, n_outcome, n_hidden=best_param.n_outcome_rep + best_param.n_confounder_rep
    )
    # load best iteration
    _, model_file = load_model(model, model_path, model_id_to_load)
    torch.save(model, model_path + "best.pth")

    # evaluate

    with torch.no_grad():
        y_list = []
        for i in range(len(new_x_list)):
            new_x = new_x_list[i]
            y_hat = model(new_x).cpu().numpy()
            y_list.append(y_hat)

    with torch.no_grad():

        y_mat_true = np.concatenate(dg.outcome_list, axis=-1)
        # N, 2^K
        y_mat = np.concatenate(y_list, axis=-1)
        n_test = y_mat.shape[0]
        err_all = np.sum((y_mat_true[-n_test:, :] - y_mat) ** 2, axis=1)
        rmse_all = np.sqrt(np.mean(err_all))
        rmse_all_sd = bootstrap_RMSE(torch.tensor(err_all))  # pylint: disable=not-callable

        y_mean = np.mean(y_mat_true, axis=0)[None, :]
        err_mean = np.sum((y_mat_true[-n_test:, :] - y_mean) ** 2, axis=1)
        rmse_mean = np.sqrt(np.mean(err_mean))
        print(round(rmse_mean, 3))

        print(0, 0, round(rmse_all, 3), round(rmse_all_sd, 3))


if __name__ == "__main__":
    # config1 = sim_config.sim_dict['n_confounder_10_linear']
    # config1 = sim_config.sim_dict['real_3000']
    # run(config1, eval_only=True)

    parser = argparse.ArgumentParser("EA")
    parser.add_argument("--p", type=float, default=0)
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--config", type=str)
    parser.add_argument("--revert", type=str, default="n")
    parser.add_argument("--fb", type=bool, default=False)

    args = parser.parse_args()

    config_key = args.config
    config1 = sim_config.sim_dict[config_key]

    run(config1, args.p, args.sigma, eval_only=False, revert=args.revert, full_balance=args.fb)
