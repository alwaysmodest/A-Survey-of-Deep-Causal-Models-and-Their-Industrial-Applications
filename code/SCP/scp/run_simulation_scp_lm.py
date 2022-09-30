import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV

import sim_config
from mlp import DirectOutcomeRegression, ModelTrainer
from sim_data_gen import DataGenerator
from utils import bootstrap_RMSE


def get_scp_data(dataset, single_cause_index, model, data_gen=None, oracle=False, is_train=None):
    x_train = dataset.tensors[0].detach().cpu().numpy()
    x_train[:, single_cause_index] = 1 - x_train[:, single_cause_index]

    y_train = model.predict(x_train)

    return (
        torch.tensor(  # pylint: disable=not-callable
            x_train, device=dataset.tensors[0].device, dtype=dataset.tensors[0].dtype
        ),
        torch.tensor(  # pylint: disable=not-callable
            y_train, device=dataset.tensors[0].device, dtype=dataset.tensors[0].dtype
        ),
    )


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
    linear_model=False,
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

    max_epoch = 100
    # max_epoch = 5000
    model_id = "SCP" if not ablate.is_ablate else "SCP-{}".format(ablate.ablation_id)

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

    rmse = nn.MSELoss()

    for single_cause_index in range(n_confounder, min(n_confounder + n_cause, n_confounder + ablate.perturb_subset)):

        cause_index_zero = single_cause_index - n_confounder

        # train single cause po model

        cause_logit = dg.cause_logit[: dg.train_size, cause_index_zero]
        cause = dg.cause[: dg.train_size, cause_index_zero]
        cause_prob = 1.0 / (1.0 + np.exp(-1.0 * cause_logit))
        cause_prob[cause == 0] = 1.0 - cause_prob[cause == 0]
        weight = 1.0 / cause_prob

        # train model

        model = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

        model.fit(x_train_scp_list[0].cpu().numpy(), y_train_scp_list[0].cpu().numpy(), weight)

        # generate augmented dataset
        # here flip one cause
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
    model = DirectOutcomeRegression(
        n_confounder, n_cause, n_outcome, n_hidden=n_confounder + n_cause + 1, linear=linear_model
    )
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
    parser.add_argument("--linear_model", type=bool, default=False)

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
        linear_model=args.linear_model,
    )
