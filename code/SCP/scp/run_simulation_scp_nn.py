import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn

import sim_config
from mlp import DirectOutcomeRegression, ModelTrainer
from sim_data_gen import DataGenerator
from utils import (
    NN_SCP,
    bootstrap_RMSE,
    create_NN_SCP,
    get_scp_config,
    get_scp_config_one,
    load_model,
)


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


def run(
    d_config,
    ablate=None,
    hyper_param_itr=5,
    train_ratio=None,
    eval_only=False,
    eval_delta=False,
    save_data=False,
    seed=None,
    no_confounder=False,
):
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
    sample_size_train = d_config.sample_size_train

    n_flip = d_config.n_flip

    # use default ablation if not provided (no ablation)
    if ablate is None:
        ablate = sim_config.AblationConfig()

    max_epoch = 5000
    model_id = "SCP-NN" if not ablate.is_ablate else "SCP-NN-{}".format(ablate.ablation_id)

    model_path = "model/{}_{}_model/".format(model_id, d_config.sim_id)
    if not eval_only:
        try:
            shutil.rmtree(model_path)
        except OSError as e:
            print("shutil note: %s - %s." % (e.filename, e.strerror))

    if seed is None:
        seed = 100

    if train_ratio is None:
        if d_config.real_data:
            train_ratio = 1
        else:
            train_ratio = sample_size / 1000

    np.random.seed(seed)
    torch.manual_seed(seed)

    if sample_size_train == 0:

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
            no_confounder=no_confounder,
        )
    else:
        valid_sample_size = 200
        eval_sample_size = 4100
        train_sample_size = sample_size_train
        sample_size = train_sample_size + valid_sample_size + eval_sample_size
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
            train_frac=train_sample_size / sample_size,
            val_frac=valid_sample_size / sample_size,
            p_outcome_single=p_outcome_single,
            p_outcome_double=p_outcome_double,
            outcome_interaction=outcome_interaction,
            no_confounder=no_confounder,
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

    if not no_confounder:
        param_list = get_scp_config(hyper_param_itr, n_confounder, p_confounder_cause)
    else:
        param_list = get_scp_config_one(hyper_param_itr, n_confounder, n_cause)
    rmse = nn.MSELoss()

    for single_cause_index in range(n_confounder, min(n_confounder + n_cause, n_confounder + ablate.perturb_subset)):

        err_list = list()
        # hyper-parameter search
        for param in param_list:
            # train single cause po model
            model_id_to_save = model_id + "_cause_{}_po_itr_{}".format(single_cause_index, param.itr)
            # todo
            model = create_NN_SCP(single_cause_index, d_config, param)
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
                # propensity = torch.exp(log_propensity)[:, 1:2]
                # print(treated_label)
                # print(treated_label.shape)

                # weight = (treated_label * 1. - propensity) / (1. - propensity + 1E-9) / (propensity + 1E-9)
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
        model = create_NN_SCP(single_cause_index, d_config, best_param)

        # load best iteration
        _, model_file = load_model(model, model_path, model_id_to_load)
        shutil.copy(model_path + model_file, model_path + model_id_best)

        # train potential cause model
        if p_cause_cause > 0 and not ablate.exact_single_cause:
            print("Training potential cause model")
            k_cause = single_cause_index - n_confounder
            res = dg.generate_dataset_potential_cause(k_cause, predict_all_causes=ablate.predict_all_causes)
            if res is not None:
                train_dataset_pcc, valid_dataset_pcc, x_test_pcc, y_test_pcc = res
                print(x_test_pcc.shape)
                print(y_test_pcc.shape)

                # need to redeclare the parameters
                model_id_pcc = model_id + "_potential_cause_{}".format(k_cause)
                single_cause_index_pcc = x_test_pcc.shape[1] - 1
                n_confounder_pcc = x_test_pcc.shape[1] - 1
                n_cause_pcc = 1
                n_outcome_pcc = y_test_pcc.shape[1]

                model_pcc = NN_SCP(
                    single_cause_index_pcc,
                    n_confounder_pcc,
                    n_cause_pcc,
                    n_outcome_pcc,
                    n_confounder_rep=int(n_confounder * p_confounder_cause),
                    n_outcome_rep=int(n_confounder * (1 - p_confounder_cause)) + 1,
                    mmd_sigma=1.0,
                    lam_factual=1.0,
                    lam_propensity=1.0,
                    lam_mmd=0.0,
                    binary_outcome=True,
                )
                optimizer = torch.optim.Adam(model_pcc.parameters(), lr=0.005)

                trainer = ModelTrainer(100, max_epoch, model_pcc.loss, model_id_pcc, model_path)

                trainer.train(model_pcc, optimizer, train_dataset_pcc, valid_dataset_pcc)
                model_pcc, _ = load_model(model_pcc, model_path, model_id_pcc)

                with torch.no_grad():
                    # 1. predict potential causes
                    x_all, y_all = dg.generate_dataset_potential_cause(
                        k_cause, return_dataset=False, predict_all_causes=ablate.predict_all_causes
                    )
                    x_all[:, single_cause_index_pcc] = 1 - x_all[:, single_cause_index_pcc]
                    y_hat_prob = model_pcc(x_all)[0]
                    potential_cause = torch.bernoulli(y_hat_prob)

                    # 2. get new x for single cause po
                    # n_confounder + n_cause
                    # todo: oracle potential cause
                    if ablate.oracle_potential_cause:
                        x_single_cause_po = dg.get_x_potential_cause_oracle(k_cause)
                    else:
                        x_single_cause_po = dg.get_x_potential_cause(
                            k_cause, potential_cause, predict_all_causes=ablate.predict_all_causes
                        )

                    # 3. get new y for single cause po
                    # todo: oracle model
                    y_single_cause_po = get_scp_data_potential_cause(x_single_cause_po, model, dg, ablate.oracle_po)

                    # 4. split data
                    train_dataset_new, valid_dataset_new, _, _ = dg.split_xy(x_single_cause_po, y_single_cause_po)

                    x_train_scp, y_train_scp = train_dataset_new.tensors  # pylint: disable=unbalanced-tuple-unpacking
                    x_valid_scp, y_valid_scp = valid_dataset_new.tensors  # pylint: disable=unbalanced-tuple-unpacking

            else:
                # generate augmented dataset
                # here only flip one cause (due to lack of descendants)
                x_train_scp, y_train_scp = get_scp_data(
                    train_dataset, single_cause_index, model, dg, ablate.oracle_po, True
                )
                x_valid_scp, y_valid_scp = get_scp_data(valid_dataset, single_cause_index, model, dg, ablate.oracle_po)
        else:
            # generate augmented dataset
            # here only flip one cause (due to ablation: ablate.exact_single_cause == True)
            x_train_scp, y_train_scp = get_scp_data(train_dataset, single_cause_index, model)
            x_valid_scp, y_valid_scp = get_scp_data(valid_dataset, single_cause_index, model)

        x_train_scp_list.append(x_train_scp)
        y_train_scp_list.append(y_train_scp)
        x_valid_scp_list.append(x_valid_scp)
        y_valid_scp_list.append(y_valid_scp)

    # create aggregated dataset
    print(len(x_train_scp_list))

    x_train = torch.cat(x_train_scp_list, dim=0)
    y_train = torch.cat(y_train_scp_list, dim=0)
    x_valid = torch.cat(x_valid_scp_list, dim=0)
    y_valid = torch.cat(y_valid_scp_list, dim=0)

    if save_data:
        torch.save(x_train, "{}_{}_{}_x.pth".format(config_key, model_id, seed))
        return 0

    train_dataset = torch.utils.data.dataset.TensorDataset(x_train, y_train)
    valid_dataset = torch.utils.data.dataset.TensorDataset(x_valid, y_valid)

    # train model on aggregated dataset
    batch_size = 100
    model = DirectOutcomeRegression(n_confounder, n_cause, n_outcome, n_hidden=n_confounder + n_cause + 1, linear=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    trainer = ModelTrainer(batch_size, max_epoch, rmse, model_id, model_path)

    trainer.train(model, optimizer, train_dataset, valid_dataset, print_every=10)
    torch.save(model, model_path + "best.pth")

    # test on hold-out test data

    if eval_delta:
        for j in range(n_cause):
            n_flip = j + 1
            new_x_test, cate_test = dg.generate_counterfactual_test(n_flip)
            with torch.no_grad():
                y_hat0 = model(x_test)
                y_hat1 = model(new_x_test)
                cate_hat = y_hat1 - y_hat0
                error = torch.sqrt(rmse(cate_hat, cate_test))
                rmse_sd = bootstrap_RMSE((cate_hat - cate_test) ** 2)
                print("scp", n_flip, round(error.item(), 3), round(rmse_sd, 3))
        return 0

    with torch.no_grad():
        y_list = []
        for i in range(len(new_x_list)):
            new_x = new_x_list[i]
            y_hat = model(new_x).cpu().numpy()
            y_list.append(y_hat)

    if not d_config.real_data:
        with torch.no_grad():

            y_hat0 = model(x_test)
            y_hat1 = model(new_x_test)
            cate_hat = y_hat1 - y_hat0
            error = torch.sqrt(rmse(cate_hat, cate_test))
            rmse_sd = bootstrap_RMSE((cate_hat - cate_test) ** 2)

            y_mat_true = np.concatenate(dg.outcome_list, axis=-1)
            # N, 2^K
            y_mat = np.concatenate(y_list, axis=-1)
            n_test = y_mat.shape[0]
            err_all = np.sum((y_mat_true[-n_test:, :] - y_mat) ** 2, axis=1)
            rmse_all = np.sqrt(np.mean(err_all))
            rmse_all_sd = bootstrap_RMSE(torch.tensor(err_all))  # pylint: disable=not-callable

            print(round(error.item(), 3), round(rmse_sd, 3), round(rmse_all, 3), round(rmse_all_sd, 3))
    else:
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
    parser.add_argument("--save_data", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_confounder", type=bool, default=False)

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

    run(
        config1,
        ablate=abl_config,
        hyper_param_itr=1,
        train_ratio=5.0,
        save_data=args.save_data,
        seed=args.seed,
        no_confounder=args.no_confounder,
    )
