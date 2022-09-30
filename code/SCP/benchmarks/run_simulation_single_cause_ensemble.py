import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn

import sim_config
from mlp import ModelTrainer
from sim_data_gen import DataGenerator
from utils import bootstrap_RMSE, create_DR_CRN, get_scp_config, load_model


def get_scp_data(dataset, single_cause_index, model, data_gen=None, oracle=False, is_train=None):
    with torch.no_grad():
        x_train = dataset.tensors[0].detach().clone()
        x_train[:, single_cause_index] = 1 - x_train[:, single_cause_index]

    if not oracle:
        with torch.no_grad():
            y_train = model(x_train)[0]
    else:
        print("Use Oracle")
        causes = x_train[:, data_gen.n_confounder :].cpu().numpy()

        if is_train:
            pad = np.zeros((data_gen.sample_size - causes.shape[0], data_gen.n_cause))
            causes_total = np.concatenate((causes, pad))
        else:
            pad1 = np.zeros((data_gen.train_size, data_gen.n_cause))
            pad2 = np.zeros((data_gen.sample_size - data_gen.train_size - data_gen.val_size, data_gen.n_cause))
            causes_total = np.concatenate((pad1, causes, pad2))

        y_train_total = data_gen.generate_counterfactual(causes_total)

        if is_train:
            y_train = y_train_total[: data_gen.train_size, :]
        else:
            y_train = y_train_total[data_gen.train_size : data_gen.train_size + data_gen.val_size, :]

        y_train = torch.tensor(y_train).to(x_train)  # pylint: disable=not-callable
    return x_train, y_train


def get_scp_data_potential_cause(x_train, model, data_gen=None, oracle=False):
    if not oracle:
        with torch.no_grad():
            y_train = model(x_train)[0]
    else:
        print("Use Oracle")
        assert data_gen is not None
        x_train1 = x_train.cpu().numpy()
        causes = x_train1[:, data_gen.n_confounder :]
        y_train = data_gen.generate_counterfactual(causes)
        y_train = torch.tensor(y_train).to(x_train)  # pylint: disable=not-callable
    return y_train


def run(d_config, ablate=None, hyper_param_itr=5, train_ratio=None, eval_only=False):
    n_confounder = d_config.n_confounder
    n_cause = d_config.n_cause
    n_outcome = d_config.n_outcome
    sample_size = d_config.sample_size
    p_confounder_cause = d_config.p_confounder_cause
    p_cause_cause = d_config.p_cause_cause
    cause_noise = d_config.cause_noise
    outcome_noise = d_config.outcome_noise
    linear = d_config.linear
    p_outcome_single = d_config.p_outcome_single
    p_outcome_double = d_config.p_outcome_double
    outcome_interaction = d_config.outcome_interaction

    n_flip = d_config.n_flip

    # use default ablation if not provided (no ablation)
    if ablate is None:
        ablate = sim_config.AblationConfig()

    max_epoch = 100
    model_id = "SCP-ENS" if not ablate.is_ablate else "SCP-ENS-{}".format(ablate.ablation_id)

    model_path = "model/{}_{}_model/".format(model_id, d_config.sim_id)
    if not eval_only:
        try:
            shutil.rmtree(model_path)
        except OSError as e:
            print("shutil note: %s - %s." % (e.filename, e.strerror))

    seed = 100
    if train_ratio is None:
        train_ratio = sample_size / 1000

    np.random.seed(seed)
    torch.manual_seed(seed)

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

    print(np.mean(dg.cause, axis=0))

    # iterate over each cause

    x_train_scp_list = [train_dataset.tensors[0].detach().clone()]
    y_train_scp_list = [train_dataset.tensors[1].detach().clone()]
    x_valid_scp_list = [valid_dataset.tensors[0].detach().clone()]
    y_valid_scp_list = [valid_dataset.tensors[1].detach().clone()]

    param_list = get_scp_config(hyper_param_itr, n_confounder, p_confounder_cause)
    rmse = nn.MSELoss()
    y_hat_list = []
    cate_hat_list = []
    for single_cause_index in range(n_confounder, min(n_confounder + n_cause, n_confounder + ablate.perturb_subset)):

        err_list = list()
        # hyper-parameter search
        for param in param_list:
            # train single cause po model
            model_id_to_save = model_id + "_cause_{}_po_itr_{}".format(single_cause_index, param.itr)
            model = create_DR_CRN(single_cause_index, d_config, param)
            optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)
            if not ablate.oracle_po:
                trainer = ModelTrainer(param.batch_size, max_epoch, model.loss, model_id_to_save, model_path)
            else:
                trainer = ModelTrainer(param.batch_size, 1, model.loss, model_id_to_save, model_path)
            trainer.train(model, optimizer, train_dataset, valid_dataset, print_every=100)

            # load best iteration
            load_model(model, model_path, model_id_to_save)

            # evaluate factual loss on validation dataset
            with torch.no_grad():
                x_valid = x_valid_scp_list[0]
                y_valid = y_valid_scp_list[0]

                y_hat, log_propensity, mmd, treated_label = model(x_valid)
                propensity = torch.exp(log_propensity)[:, 1:2]
                # print(treated_label)
                # print(treated_label.shape)

                weight = (treated_label * 1.0 - propensity) / (1.0 - propensity + 1e-9) / (propensity + 1e-9)
                # print(weight)
                # print(torch.sum(torch.isnan(weight)))
                # error = torch.mean((weight * (y_hat - y_valid)) ** 2)
                #
                error = torch.sqrt(rmse(y_hat, y_valid))
                print("Validation error:", round(error.item(), 3))
                err_list.append(error)

        # select model with best hyper-parameter
        best_index = int(np.argmin(np.array(err_list)))
        best_param = param_list[best_index]
        print("Best param:", best_param)
        model_id_to_load = model_id + "_cause_{}_po_itr_{}".format(single_cause_index, best_param.itr)
        model_id_best = model_id + "_cause_{}_po.pt".format(single_cause_index)
        model = create_DR_CRN(single_cause_index, d_config, best_param)

        # load best iteration
        _, model_file = load_model(model, model_path, model_id_to_load)
        shutil.copy(model_path + model_file, model_path + model_id_best)
        y_list = []
        with torch.no_grad():
            y_hat0 = model(x_test)[0]
            y_hat1 = model(new_x_test)[0]
            cate_hat = y_hat1 - y_hat0
            cate_hat_list.append(cate_hat)
            for i in range(len(new_x_list)):
                new_x = new_x_list[i]
                y_hat = model(new_x)[0]
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
    # config1 = sim_config.sim_dict['real_3000']
    # run(config1)

    # config1 = sim_config.dependent_cause
    # abl_config = None

    # abl_config = sim_config.AblationConfig(perturb_subset=4, ablation_id='perturb_subset-4')
    # abl_config = sim_config.AblationConfig(exact_single_cause=True, ablation_id='exact_single_cause')
    # abl_config = sim_config.AblationConfig(predict_all_causes=True, ablation_id='predict_all_causes')
    # run(config1, abl_config)

    # run ablation
    parser = argparse.ArgumentParser("Ablation")
    parser.add_argument("--ablation", type=str, default="None")
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    i = args.ablation.find("perturb_subset")
    if i >= 0:
        subset = int(args.ablation.split("-")[-1])
        ablation = args.ablation[:i]
    else:
        subset = 10000
        ablation = args.ablation

    if ablation == "None":
        abl_config = None
    elif ablation == "exact_single_cause":
        abl_config = sim_config.AblationConfig(
            exact_single_cause=True, perturb_subset=subset, ablation_id="exact_single_cause"
        )
    elif ablation == "predict_all_causes":
        abl_config = sim_config.AblationConfig(
            predict_all_causes=True, perturb_subset=subset, ablation_id="predict_all_causes"
        )
    elif ablation == "oracle_po":
        abl_config = sim_config.AblationConfig(oracle_po=True, perturb_subset=subset, ablation_id="oracle_po")
    elif ablation == "oracle_potential_cause":
        abl_config = sim_config.AblationConfig(
            oracle_potential_cause=True, perturb_subset=subset, ablation_id="oracle_potential_cause"
        )
    elif ablation == "oracle_all":
        abl_config = sim_config.AblationConfig(
            oracle_po=True, perturb_subset=subset, oracle_potential_cause=True, ablation_id="oracle_all"
        )
    elif args.ablation.find("perturb_subset") >= 0:
        subset = int(args.ablation.split("-")[-1])
        abl_config = sim_config.AblationConfig(perturb_subset=subset, ablation_id=args.ablation)
    else:
        exit(-1)

    config_key = args.config

    try:
        config1 = sim_config.sim_dict[config_key]
    except KeyError:
        print(config_key)
        exit(-1)

    run(config1, ablate=abl_config, hyper_param_itr=1, train_ratio=5.0)
