from typing import NamedTuple

import numpy as np
import pandas as pds
import torch
from sklearn.decomposition import PCA

from global_config import DEVICE, DTYPE
from utils import bootstrap_RMSE


def logistic(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def cause_to_num(cause):
    # convert the causes into an index
    weight_vector = np.power(2, np.arange(0, cause.shape[1]))
    cause_ind = np.matmul(cause, weight_vector)
    return cause_ind


# def num_to_cause(num, n_treatment):
#     num = torch.tensor(num)
#     cause = num.unsqueeze(-1).to(torch.long)
#     cause_mat = torch.zeros(cause.shape[0], n_treatment)
#     cause_mat.scatter_(1, cause, 1)
#


def dcg_at_k(r, k):
    r = r[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    # ndcg_at_k(score_mat, k)
    dcg_max = dcg_at_k(np.array(sorted(r[0], reverse=True)), k)

    res = []
    for i in range(r.shape[0]):
        res.append(dcg_at_k(r[i], k) / dcg_max)
    return np.array(res)


class DataGeneratorConfig(NamedTuple):
    n_confounder: int
    n_cause: int
    n_outcome: int
    sample_size: int
    p_confounder_cause: float
    p_cause_cause: float
    cause_noise: float
    outcome_noise: float
    linear: bool
    sim_id: str
    p_outcome_single: float = 1.0
    p_outcome_double: float = 0.0
    train_frac: float = 0.7
    val_frac: float = 0.1
    n_flip: int = 1
    confounding_level: float = 1.0
    real_data: bool = False
    outcome_interaction: bool = False
    sample_size_train: int = 0


class DataGenerator:
    def __init__(
        self,
        n_confounder,
        n_cause,
        n_outcome,
        sample_size,
        p_confounder_cause,
        p_cause_cause,
        cause_noise,
        outcome_noise,
        linear,
        p_outcome_single=1,
        p_outcome_double=1,
        confounding_level=1.0,
        train_frac=0.7,
        val_frac=0.1,
        real_data=False,
        outcome_interaction=False,
        outcome_interaction3=False,
        no_confounder=False,
        device=DEVICE,
        dtype=DTYPE,
    ):
        assert train_frac + val_frac <= 1
        self.n_confounder = n_confounder
        self.n_cause = n_cause
        self.n_outcome = n_outcome
        self.sample_size = sample_size
        self.p_confounder_cause = p_confounder_cause
        self.p_cause_cause = p_cause_cause
        self.cause_noise = cause_noise
        self.outcome_noise = outcome_noise
        self.linear = linear
        self.outcome_interaction = outcome_interaction
        self.outcome_interaction3 = outcome_interaction3
        self.p_outcome_single = p_outcome_single
        self.p_outcome_double = p_outcome_double
        self.confounder = None
        self.coefficient_confounder_cause = None
        self.coefficient_cause_cause = None
        self.coefficient_cause_outcome = None
        self.coefficient_confounder_outcome = None
        self.cause = None
        self.outcome = None
        self.outcome_error = None
        self.cause_error = None
        self.cause_logit = None
        self.train_size = int(train_frac * sample_size)
        self.val_size = int(val_frac * sample_size)
        self.device = device
        self.dtype = dtype
        self.confounding_level = confounding_level
        self.generated = False
        self.descendant = None
        self.real_data = real_data
        self.coef_list = None
        self.outcome_list = None
        self.cause_ind = None
        self.ranks = None
        self.interaction_coef = None
        self.pca = None
        self.no_confounder = no_confounder

        # iterate over causes
        n_treatment = int(np.power(2, self.n_cause))
        # 2^K, K
        cause_ref = np.zeros((n_treatment, self.n_cause))
        for i in range(n_treatment):
            tmp = i
            for j in range(1, self.n_cause + 1):
                value = tmp % 2
                tmp = tmp // 2
                cause_ref[i, j - 1] = value

        self.cause_ref = cause_ref
        self.n_treatment = n_treatment

    def _make_tensor(self, x):
        if type(x) == np.ndarray:
            return torch.tensor(x, dtype=self.dtype, device=self.device)  # pylint: disable=not-callable
        else:
            return x.to(dtype=self.dtype, device=self.device)

    def get_coefficient_confounder_cause(self):
        coef = np.random.randn(self.n_confounder, self.n_cause)
        mask = np.random.binomial(1, self.p_confounder_cause, (self.n_confounder, self.n_cause))
        if not self.no_confounder:
            self.coefficient_confounder_cause = coef * mask
        else:
            self.coefficient_confounder_cause = coef * 0.0

    def get_coefficient_cause_cause(self):
        coef = np.random.randn(self.n_cause, self.n_cause)
        mask = np.random.binomial(1, self.p_cause_cause, (self.n_cause, self.n_cause))
        self.coefficient_cause_cause = coef * mask

        children_list = [list() for i in range(self.n_cause)]
        for i in reversed(range(self.n_cause)):
            children = list(np.where(mask[i, min(i + 1, self.n_cause) :])[0] + i + 1)

            while True:
                set_card = len(children)
                for j in children:
                    children = children + children_list[j]

                children = list(set(children))
                new_set_card = len(children)
                if new_set_card == set_card:
                    break

            children_list[i] = children

        self.descendant = children_list

    def get_coefficient_outcome(self):
        coef_cause = np.random.randn(self.n_cause)
        coef_confounder = np.random.randn(self.n_confounder)
        self.coefficient_cause_outcome = coef_cause
        if not self.no_confounder:
            self.coefficient_confounder_outcome = coef_confounder
        else:
            self.coefficient_confounder_outcome = coef_confounder * 0.0

    def generate_confounder(self):
        """
        Generate the confounders using i.i.d Gaussian(0, 1). Sample size x dim
        """
        if not self.no_confounder:
            self.confounder = np.random.randn(self.sample_size, self.n_confounder)
        else:
            self.confounder = np.zeros((self.sample_size, self.n_confounder)) * 1.0

    def get_one_cause(self, i, confounder_factor_mean, previous_causes):
        # K
        mean = confounder_factor_mean[:, i]

        # todo: add parameter here
        if i > 0:
            cause_factor_mean = (
                np.matmul(previous_causes, self.coefficient_cause_cause[:i, i]) / previous_causes.shape[1]
            )
        else:
            cause_factor_mean = 0
        cause_error = np.random.randn(self.sample_size) * self.cause_noise
        cause_logit = mean + cause_factor_mean + cause_error
        cause = np.random.binomial(1, logistic(cause_logit))
        return cause, cause_logit, cause_error

    def generate_cause(self):
        confounder_factor_mean = (
            np.matmul(self.confounder, self.coefficient_confounder_cause) / self.n_confounder * self.confounding_level
        )

        cause_list = []
        cause_logit_list = []
        cause_error_list = []
        for i in range(self.n_cause):
            if i > 0:
                previous_causes = np.stack(cause_list, axis=-1)
            else:
                previous_causes = None

            new_cause, cause_logit, cause_error = self.get_one_cause(i, confounder_factor_mean, previous_causes)
            cause_list.append(new_cause)
            cause_logit_list.append(cause_logit)
            cause_error_list.append(cause_error)

        self.cause = np.stack(cause_list, axis=-1)
        self.cause_logit = np.stack(cause_logit_list, axis=-1)
        self.cause_error = np.stack(cause_error_list, axis=-1)

    def generate_outcome(self):
        if not self.outcome_interaction:
            # no interaction
            outcome = np.matmul(self.cause, self.coefficient_cause_outcome)
            outcome = outcome + np.matmul(self.confounder, self.coefficient_confounder_outcome)

        elif self.outcome_interaction3:
            # three way interaction
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, self.cause), axis=1)
            ncol_feat = feat.shape[1]

            coef_list = []

            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    for k in range(j, ncol_feat):
                        for h in range(k, ncol_feat):

                            feat_in = feat[:, i] * feat[:, j] * feat[:, k] * feat[:, h]
                            coef = np.random.randn()

                            if i == 0:
                                mask = np.random.binomial(1, self.p_outcome_single)
                            else:
                                mask = np.random.binomial(1, self.p_outcome_double)
                            coef *= mask
                            coef_list.append(coef)

                            outcome = outcome + feat_in * coef

            self.interaction_coef = np.array(coef_list)

        else:
            # two way interaction
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, self.cause), axis=1)
            ncol_feat = feat.shape[1]

            coef_list = []

            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    feat_in = feat[:, i] * feat[:, j]
                    coef = np.random.randn()

                    if i == 0:
                        mask = np.random.binomial(1, self.p_outcome_single)
                    else:
                        mask = np.random.binomial(1, self.p_outcome_double)
                    coef *= mask
                    coef_list.append(coef)

                    outcome = outcome + feat_in * coef

            self.interaction_coef = np.array(coef_list)

        self.outcome_error = np.random.randn(self.sample_size) * self.outcome_noise
        outcome = outcome + self.outcome_error
        outcome = outcome[:, None]
        assert len(outcome.shape) == 2
        if not self.linear:
            if not self.outcome_interaction:
                outcome = logistic(outcome / np.sqrt(self.n_confounder))
            else:
                outcome = logistic((outcome - np.mean(outcome)) / np.std(outcome))
        self.outcome = outcome

    def generate_counterfactual(self, new_cause):
        if not self.outcome_interaction:
            outcome = np.matmul(new_cause, self.coefficient_cause_outcome)
            outcome = outcome + np.matmul(self.confounder, self.coefficient_confounder_outcome)

        elif self.outcome_interaction3:

            # two way interaction
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, new_cause), axis=1)
            ncol_feat = feat.shape[1]
            counter = 0

            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    for k in range(j, ncol_feat):
                        for h in range(k, ncol_feat):

                            feat_in = feat[:, i] * feat[:, j] * feat[:, k] * feat[:, h]
                            outcome = outcome + feat_in * self.interaction_coef[counter]
                            counter += 1

        else:
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, new_cause), axis=1)
            ncol_feat = feat.shape[1]

            counter = 0
            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    feat_in = feat[:, i] * feat[:, j]
                    outcome = outcome + feat_in * self.interaction_coef[counter]
                    counter += 1

        # use the old error
        outcome = outcome + self.outcome_error
        outcome = outcome[:, None]
        assert len(outcome.shape) == 2
        if not self.linear:
            outcome = logistic(outcome / np.sqrt(self.n_confounder))
        return outcome

    def generate_real_outcome(self, cause, confounder, noise=0.0):
        # feature dimension
        ones = np.ones((confounder.shape[0], 1))
        feat_dim = self.n_cause + self.n_confounder + 1
        coef_list = []
        for i in range(feat_dim):
            for j in range(i + 1, feat_dim):
                this_coef = np.random.randn() * 0.5
                coef_list.append(this_coef)
        self.coef_list = coef_list

        # iterate over causes
        n_treatment = int(np.power(2, self.n_cause))
        # 2^K, K
        cause_ref = np.zeros((n_treatment, self.n_cause))
        for i in range(n_treatment):
            tmp = i
            for j in range(1, self.n_cause + 1):
                value = tmp % 2
                tmp = tmp // 2
                cause_ref[i, j - 1] = value

        self.cause_ref = cause_ref
        # 2^K
        cause_ind = cause_to_num(cause_ref)

        # generate outcomes for all causes
        outcome_list = []
        for x in range(n_treatment):
            # 1, K
            cause_setting = cause_ref[x : x + 1, :]
            # N, K
            cause_sample = np.concatenate([cause_setting] * confounder.shape[0], axis=0)
            # N, K + D + 1
            feat = np.concatenate([confounder, cause_sample, ones], axis=-1)

            outcomes_mean = np.zeros((cause.shape[0], 1))
            loc = 0
            for i in range(feat.shape[1]):
                for j in range(i + 1, feat.shape[1]):
                    this_feat = feat[:, i : i + 1] * feat[:, j : j + 1]
                    this_coef = coef_list[loc]
                    loc += 1
                    outcomes_mean = outcomes_mean + this_feat * this_coef

            this_outcome = np.abs(outcomes_mean) + noise * np.random.exponential(1.0, outcomes_mean.shape)

            assert this_outcome.shape[0] == cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)

        self.outcome_list = outcome_list
        self.cause_ind = cause_ind

        # generate true ranking
        # smaller, better
        # N, 2^K
        outcome_tensor = np.concatenate(self.outcome_list, axis=-1)
        order = outcome_tensor.argsort(axis=-1)
        # N, 2^K
        ranks = order.argsort(axis=-1)
        self.ranks = ranks

        # select observed outcomes from the list
        # N
        obs_cause_ind = cause_to_num(cause)
        outcomes = np.zeros((cause.shape[0], 1))
        for i in range(cause.shape[0]):
            location = np.where(cause_ind == obs_cause_ind[i])[0][0]
            outcomes[i, 0] = self.outcome_list[location][i, 0]

        return outcomes

    def get_relevance_score(self, predicted_ranking):
        # predicted ranking on the testing set
        n_treatment = np.power(2, self.n_cause)

        score_list = []
        for i in range(predicted_ranking.shape[0]):
            predicted = predicted_ranking[i]
            true = self.ranks[self.train_size + self.val_size + i]
            # scores = n_treatment - true[predicted.astype('int')]
            scores = (true[predicted.astype("int")] < 5).astype("float")
            score_list.append(scores)

        # N, 2^K
        score_mat = np.stack(score_list, axis=0)
        return score_mat

    def generate_oracle_improvements(self):
        assert self.generated
        outcome_tensor = np.concatenate(self.outcome_list, axis=-1)
        best_outcome = np.min(outcome_tensor, axis=-1)
        improvements = self.outcome - best_outcome[:, None]
        return improvements[self.train_size + self.val_size :, 0]

    def get_predicted_improvements(self, order, k=5):
        pred_list = []
        for j in range(k):
            best_y_ind = order[:, j]
            score_list = []
            for i in range(best_y_ind.shape[0]):
                predicted = best_y_ind[i]
                true = self.outcome_list[predicted][self.train_size + self.val_size + i]
                score_list.append(true)

            pred = np.array(score_list)
            pred_list.append(pred)

        pred_mat = np.stack(pred_list, axis=-1)
        pred = np.mean(pred_mat, axis=-1)
        improvements = self.outcome - pred[:, None]
        return improvements[:, 0]

    def generate_real(self):
        # load data
        cause = pds.read_csv("real_data/cause.csv").values
        confounder = pds.read_csv("real_data/confounder.csv").values

        assert cause.shape == (3080, 5)
        assert confounder.shape == (3080, 17)
        #
        # tv_sample_size = self.sample_size
        #
        # self.sample_size = confounder.shape[0]
        # self.train_size = int(tv_sample_size * 0.9)
        # self.val_size = int(tv_sample_size * 0.1)

        outcomes = self.generate_real_outcome(cause, confounder, noise=0.1)
        self.cause = cause
        self.confounder = confounder
        self.outcome = outcomes
        return outcomes

    def generate(self):
        if not self.generated:
            if self.real_data:
                self.generate_real()
            else:
                self.get_coefficient_confounder_cause()
                self.get_coefficient_cause_cause()
                self.get_coefficient_outcome()

                self.generate_confounder()
                self.generate_cause()
                self.generate_outcome()
        self.generated = True
        return self.confounder, self.cause, self.outcome

    def split_xy(self, x, y):
        x_train = x[: self.train_size]
        y_train = y[: self.train_size]

        x_val = x[self.train_size : self.train_size + self.val_size]
        y_val = y[self.train_size : self.train_size + self.val_size]

        x_test = x[self.train_size + self.val_size :]
        y_test = y[self.train_size + self.val_size :]

        train_dataset = torch.utils.data.dataset.TensorDataset(self._make_tensor(x_train), self._make_tensor(y_train))
        valid_dataset = torch.utils.data.dataset.TensorDataset(self._make_tensor(x_val), self._make_tensor(y_val))
        x_test = self._make_tensor(x_test)
        y_test = self._make_tensor(y_test)
        return train_dataset, valid_dataset, x_test, y_test

    def generate_dataset(self, return_dataset=True, weight=None):
        self.generate()

        if weight is None:
            x = np.concatenate((self.confounder, self.cause), axis=-1)
        else:
            x = np.concatenate((self.confounder, self.cause, weight), axis=-1)
        y = self.outcome

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return self._make_tensor(x), self._make_tensor(y)

    def generate_dataset_bmc(self, return_dataset=True, weight=None, npc=3):
        self.generate()

        pca = PCA(n_components=npc)
        self.pca = pca
        pc = pca.fit_transform(self.cause)

        if weight is None:
            x = np.concatenate((self.confounder, pc, self.cause), axis=-1)
        else:
            x = np.concatenate((self.confounder, pc, self.cause, weight), axis=-1)
        y = self.outcome

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return self._make_tensor(x), self._make_tensor(y)

    def generate_dataset_tarnet(self, return_dataset=True):
        self.generate()

        cause = self._make_tensor(self.cause)
        weight_vector = torch.pow(2, torch.arange(0, self.n_cause))
        cause_ind = torch.matmul(cause, weight_vector.to(cause)).unsqueeze(-1)

        x = torch.cat([self._make_tensor(self.confounder), cause_ind], dim=1)
        y = self._make_tensor(self.outcome)

        if return_dataset:
            # x: n_confounder + 1
            # y: 1
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_propensity(self, return_dataset=True):
        self.generate()

        x = self._make_tensor(self.confounder)
        cause = self._make_tensor(self.cause)
        weight_vector = torch.pow(2, torch.arange(0, self.n_cause))
        y = torch.matmul(cause, weight_vector.to(cause)).squeeze().to(torch.long)

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_vae(self, return_dataset=True):
        self.generate()
        x = self._make_tensor(self.cause)
        y = x.detach().clone()

        if return_dataset:
            # x: K_cause
            # y: k_cause
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_dr(self, z, z_rand, return_dataset=True, shuffle=True):
        self.generate()

        x1 = torch.cat([self._make_tensor(self.confounder), z], dim=-1)
        x0 = torch.cat([self._make_tensor(self.confounder), z_rand], dim=-1)
        x = torch.cat([x1, x0], dim=0)
        y = torch.cat([torch.ones(x1.shape[0]), torch.zeros(x1.shape[0])]).unsqueeze(-1).to(x1)

        if shuffle:
            ind = torch.randperm(x.shape[0])
            x = x[ind]
            y = y[ind]

        if return_dataset:
            # x: K_cause
            # y: k_cause
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_potential_cause(self, k_cause, new_cause=None, return_dataset=True, predict_all_causes=False):
        # k_cause in [0, n_cause - 1)
        # assert k_cause != self.n_cause - 1
        self.generate()

        if new_cause is None:
            cause = self.cause
        else:
            cause = new_cause

        single_cause = cause[:, k_cause : (k_cause + 1)]

        ancestor_cause = cause[:, :k_cause]
        try:
            descendent_cause = cause[:, self.descendant[k_cause]]
        except TypeError:
            return None

        if descendent_cause.shape[1] == 0:
            return None

        if ancestor_cause.shape == 2:
            x = np.concatenate((self.confounder, ancestor_cause, single_cause), axis=-1)
        else:
            x = np.concatenate((self.confounder, single_cause), axis=-1)

        y = descendent_cause

        if predict_all_causes:
            # override all previous settings
            x = np.concatenate((self.confounder, single_cause), axis=-1)
            y = np.delete(cause, k_cause, axis=1)

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return self._make_tensor(x), self._make_tensor(y)

    #
    #
    # def get_one_cause(self, i, confounder_factor_mean, previous_causes):
    #     # K
    #     mean = confounder_factor_mean[:, i]
    #
    #     if i > 0:
    #         cause_factor_mean = np.matmul(previous_causes, self.coefficient_cause_cause[:i, i]) / previous_causes.shape[1]
    #     else:
    #         cause_factor_mean = 0
    #     cause_error = np.random.randn(self.sample_size) * self.cause_noise
    #     cause_logit = mean + cause_factor_mean + cause_error
    #     cause = np.random.binomial(1, logistic(cause_logit))
    #     return cause, cause_logit, cause_error
    #
    # def generate_cause(self):
    #     confounder_factor_mean = np.matmul(self.confounder, self.coefficient_confounder_cause) / self.n_confounder * self.confounding_level
    #
    #     cause_list = []
    #     cause_logit_list = []
    #     cause_error_list = []
    #     for i in range(self.n_cause):
    #         if i > 0:
    #             previous_causes = np.stack(cause_list, axis=-1)
    #         else:
    #             previous_causes = None
    #
    #         new_cause, cause_logit, cause_error = self.get_one_cause(i, confounder_factor_mean, previous_causes)
    #         cause_list.append(new_cause)
    #         cause_logit_list.append(cause_logit)
    #         cause_error_list.append(cause_error)
    #
    #     self.cause = np.stack(cause_list, axis=-1)
    #     self.cause_logit = np.stack(cause_logit_list, axis=-1)
    #     self.cause_error = np.stack(cause_error_list, axis=-1)
    #
    #
    #
    def get_x_potential_cause_oracle(self, k_cause, return_cause=False):
        print("Use Oracle")
        confounder_factor_mean = (
            np.matmul(self.confounder, self.coefficient_confounder_cause) / self.n_confounder * self.confounding_level
        )

        cause_list = []
        for i in range(self.n_cause):

            if i < k_cause:
                cause_list.append(self.cause[:, i])
            elif i == k_cause:
                cause_list.append(1.0 - self.cause[:, i])
            else:
                previous_causes = np.stack(cause_list, axis=-1)
                mean = confounder_factor_mean[:, i]
                cause_factor_mean = (
                    np.matmul(previous_causes, self.coefficient_cause_cause[:i, i]) / previous_causes.shape[1]
                )
                cause_error = self.cause_error[:, i]
                cause_logit = mean + cause_factor_mean + cause_error
                new_cause = np.random.binomial(1, logistic(cause_logit))
                cause_list.append(new_cause)
        cause = np.stack(cause_list, axis=-1)
        x = np.concatenate((self.confounder, cause), axis=-1)
        if return_cause:
            return cause, self._make_tensor(x)
        else:
            return self._make_tensor(x)

    def get_x_potential_cause(self, k_cause, potential_cause, predict_all_causes=False):
        if not predict_all_causes:
            potential_cause = potential_cause.cpu().numpy()
            cause = self.cause.copy()

            cause[:, k_cause] = 1.0 - cause[:, k_cause]

            for i, j in enumerate(self.descendant[k_cause]):
                cause[:, j] = potential_cause[:, i]
        else:
            potential_cause = potential_cause.cpu().numpy()
            cause = self.cause.copy()

            for i in range(cause.shape[1]):
                if i == k_cause:
                    cause[:, k_cause] = 1.0 - cause[:, k_cause]
                elif i < k_cause:
                    cause[:, i] = potential_cause[:, i]
                else:
                    cause[:, i] = potential_cause[:, i - 1]

        x = np.concatenate((self.confounder, cause), axis=-1)
        return self._make_tensor(x)

    def generate_counterfactual_test(self, n_flip, weight=None):

        new_cause = self.flip_cause(n_flip)
        new_outcome = self.generate_counterfactual(new_cause)
        new_x = np.concatenate((self.confounder, new_cause), axis=-1)

        cate_test = self._make_tensor((new_outcome - self.outcome)[self.train_size + self.val_size :])
        new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])
        if weight is not None:
            new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

        # todo: generate all counterfactual outcomes
        outcome_list = []
        for x in range(self.n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            new_cause = np.concatenate([cause_setting] * self.sample_size, axis=0)

            this_outcome = self.generate_counterfactual(new_cause)

            assert this_outcome.shape[0] == self.cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)
        self.outcome_list = outcome_list
        return new_x_test, cate_test

    def generate_counterfactual_test_bmc(self, n_flip, weight=None):

        new_cause = self.flip_cause(n_flip)
        pc = self.pca.transform(new_cause)
        new_outcome = self.generate_counterfactual(new_cause)
        new_x = np.concatenate((self.confounder, pc, new_cause), axis=-1)

        cate_test = self._make_tensor((new_outcome - self.outcome)[self.train_size + self.val_size :])
        new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])
        if weight is not None:
            new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

        # todo: generate all counterfactual outcomes
        outcome_list = []
        for x in range(self.n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            new_cause = np.concatenate([cause_setting] * self.sample_size, axis=0)

            this_outcome = self.generate_counterfactual(new_cause)

            assert this_outcome.shape[0] == self.cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)
        self.outcome_list = outcome_list
        return new_x_test, cate_test

    def generate_test_real(self, weight=None):
        new_x_list = []

        n_treatment = int(np.power(2, self.n_cause))

        for x in range(n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            # N, K
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)
            # N, K + D + 1
            new_x = np.concatenate((self.confounder, new_cause), axis=-1)

            new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])

            if weight is not None:
                new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

            new_x_list.append(new_x_test)
        return new_x_list

    def generate_test_real_bmc(self, weight=None):
        new_x_list = []

        n_treatment = int(np.power(2, self.n_cause))

        for x in range(n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            # N, K
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)

            pc = self.pca.transform(new_cause)
            # N, K + D + 1
            new_x = np.concatenate((self.confounder, pc, new_cause), axis=-1)

            new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])

            if weight is not None:
                new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

            new_x_list.append(new_x_test)
        return new_x_list

    def generate_test_tarnet(self, weight=None):
        new_x_list = []

        n_treatment = int(np.power(2, self.n_cause))

        for x in range(n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            # N, K
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)

            cause_ind = cause_to_num(new_cause)[:, None]
            # N, K + D + 1
            new_x = np.concatenate((self.confounder, cause_ind), axis=-1)

            new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])

            if weight is not None:
                new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

            new_x_list.append(new_x_test)
        return new_x_list

    def evaluate_real(self, y_list):

        y_mat_true = np.concatenate(self.outcome_list, axis=-1)

        # N, 2^K
        y_mat = np.concatenate(y_list, axis=-1)

        # PEHE
        tau_hat = y_mat[:, 1:] - y_mat[:, :-1]
        n_test = tau_hat.shape[0]
        tau = y_mat_true[-n_test:, 1:] - y_mat_true[-n_test:, :-1]

        pehe = np.sqrt(np.mean((tau - tau_hat) ** 2))
        pehe_sd = bootstrap_RMSE((torch.tensor((tau - tau_hat).flatten()) ** 2))  # pylint: disable=not-callable

        rmse = np.sqrt(np.mean((y_mat_true[-n_test:, :] - y_mat) ** 2))
        rmse_sd = bootstrap_RMSE(
            (torch.tensor((y_mat_true[-n_test:, :] - y_mat).flatten()) ** 2)  # pylint: disable=not-callable
        )

        # NDCG
        order = y_mat.argsort(axis=-1)
        # order2 = order.copy()
        # N, 2^K
        ranks_predicted = order.argsort(axis=-1)
        rel = self.get_relevance_score(ranks_predicted)
        scores = ndcg_at_k(rel, 5)
        mean_ndcg = np.mean(scores)
        sd_ndcg = np.std(scores) / np.sqrt(scores.shape[0])

        # ranking dist
        rank_dist = np.sum(np.abs(self.ranks[self.val_size + self.train_size :, :] - ranks_predicted), axis=-1)
        rank_mean = np.mean(rank_dist)
        rank_sd = np.std(rank_dist) / np.sqrt(rank_dist.shape[0])

        # improvements
        order2 = np.argmin(y_mat, axis=-1)[:, None]
        improvements = self.get_predicted_improvements(order2, k=1)
        mean_improvements = np.median(improvements)
        sd_improvements = np.std(improvements) / np.sqrt(improvements.shape[0])

        oracle = self.generate_oracle_improvements()
        mean_oracle = np.median(oracle)
        sd_oracle = np.std(oracle) / np.sqrt(oracle.shape[0])

        print(
            round(float(mean_ndcg), 3),
            round(sd_ndcg, 3),
            round(float(mean_improvements), 3),
            round(sd_improvements, 3),
            round(float(rank_mean), 3),
            round(rank_sd, 3),
            round(float(pehe), 3),
            round(pehe_sd, 3),
            round(float(rmse), 3),
            round(rmse_sd, 3),
        )
        # print(round(float(mean_ndcg), 3), round(sd_ndcg, 3))
        # print(round(float(mean_improvements), 3), round(sd_improvements, 3))
        # print(round(float(mean_oracle), 3), round(sd_oracle, 3))
        # print(round(float(rank_mean), 3), round(rank_sd, 3))

    def generate_counterfactual_test_tarnet(self, n_flip, weight=None):

        new_cause = self._make_tensor(self.flip_cause(n_flip))

        weight_vector = torch.pow(2, torch.arange(0, self.n_cause))
        cause_ind = torch.matmul(new_cause, weight_vector.to(new_cause)).unsqueeze(-1)

        x = torch.cat([self._make_tensor(self.confounder), cause_ind], dim=1)

        new_outcome = self.generate_counterfactual(new_cause.cpu().numpy())

        cate_test = self._make_tensor((new_outcome - self.outcome)[self.train_size + self.val_size :])
        new_x_test = self._make_tensor(x[self.train_size + self.val_size :])
        if weight is not None:
            new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

        # generate all counterfactual outcomes
        outcome_list = []
        for x in range(self.n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)

            this_outcome = self.generate_counterfactual(new_cause)

            assert this_outcome.shape[0] == self.cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)
        self.outcome_list = outcome_list
        return new_x_test, cate_test

    def flip_cause(self, n_flip):
        flip_index = [np.random.choice(self.n_cause, n_flip, False) for x in range(self.sample_size)]
        flip_index = np.stack(flip_index, axis=0)
        # print('flip_index', flip_index)
        flip_onehot = np.zeros((self.sample_size, self.n_cause))
        for i in range(flip_index.shape[1]):
            tmp = np.zeros((self.sample_size, self.n_cause))
            tmp[np.arange(self.sample_size), flip_index[:, i]] = 1
            flip_onehot += tmp
        new_cause = self.cause * (1 - flip_onehot) + (1.0 - self.cause) * flip_onehot
        return new_cause


if __name__ == "__main__":
    dg = DataGenerator(
        n_confounder=1,
        n_cause=2,
        n_outcome=1,
        sample_size=4,
        p_confounder_cause=0,
        p_cause_cause=1,
        cause_noise=0,
        outcome_noise=0,
        linear=True,
    )
    confounder1, cause1, outcome1 = dg.generate()
    # print(confounder)
    print(cause1)

    # print(dg.coefficient_confounder_outcome)
    # print(dg.coefficient_cause_outcome)
    # print(dg.coefficient_cause_cause)
    new_cause1 = dg.flip_cause(2)
    print(new_cause1)
    print(outcome1)
    print(dg.generate_counterfactual(new_cause1))
