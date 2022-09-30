import numpy as np
import torch
import torch.nn as nn

from global_config import DEVICE


def mmd2_rbf(Xc, Xt, sig):
    """
    https://github.com/clinicalml/cfrnet/blob/master/cfr/cfr_net.py
    Computes the l2-RBF MMD for X given t
    """
    p = 0.5

    Kcc = torch.exp(-pdist2sq(Xc, Xc) / (sig ** 2))
    Kct = torch.exp(-pdist2sq(Xc, Xt) / (sig ** 2))
    Ktt = torch.exp(-pdist2sq(Xt, Xt) / (sig ** 2))

    m = Xc.shape[0] * 1.0
    n = Xt.shape[0] * 1.0

    if m < 2 or n < 2:
        return 0.0

    mmd = ((1.0 - p) ** 2) / (m * (m - 1.0)) * (torch.sum(Kcc) - m)
    mmd = mmd + (p ** 2) / (n * (n - 1.0)) * (torch.sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * torch.sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * torch.matmul(X, Y.T)
    # print('X', X.shape)
    nx = torch.sum(X ** 2, dim=1).unsqueeze(1)
    ny = torch.sum(Y ** 2, dim=1).unsqueeze(1)
    D = (C + ny.T) + nx
    return D


class TarNet(nn.Module):
    def __init__(self, n_confounder, n_cause, n_outcome, n_hidden, mmd_sigma=0.0, lam_mmd=0.0, device=DEVICE):
        super().__init__()

        assert n_outcome == 1
        self.n_treatment = int(np.power(2, n_cause))
        self.mmd_sigma = mmd_sigma
        self.lam_mmd = lam_mmd

        self.encoder = nn.Sequential(
            nn.Linear(n_confounder, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
        ).to(device)

        heads = []

        for i in range(self.n_treatment):
            heads.append(nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)).to(device))

        self.heads = nn.ModuleList(heads)

    def forward(self, input_mat):  # pylint: disable=arguments-differ
        # confounder, cause indicator
        x = input_mat[:, :-1]
        cause = input_mat[:, -1].unsqueeze(-1).to(torch.long)
        cause_mat = torch.zeros(x.shape[0], self.n_treatment).to(x)
        cause_mat.scatter_(1, cause, 1)

        latent = self.encoder(x)

        predictions = []
        for i in range(self.n_treatment):
            y_hat = self.heads[i](latent)
            predictions.append(y_hat)

        prediction = torch.cat(predictions, dim=-1)
        prediction_selected = torch.sum(prediction * cause_mat, dim=1).unsqueeze(-1)
        if self.lam_mmd == 0.0:
            return prediction_selected
        else:
            # calculate mmd
            cause_np = cause.squeeze().cpu().numpy()
            uniques, counts = np.unique(cause_np, return_counts=True)
            uniques = uniques[counts > 1]

            if len(uniques) < 2:
                print("uniques", uniques)
                print("counts", counts)
                return prediction_selected, 0

            mmd = 0
            for i in range(len(uniques)):
                for j in range(i + 1, len(uniques)):
                    x1 = latent[cause.squeeze() == uniques[i]]
                    x2 = latent[cause.squeeze() == uniques[j]]
                    mmd = mmd + mmd2_rbf(x1, x2, self.mmd_sigma)

            return prediction_selected, mmd

    def loss(self, y_pred, y):
        # use for mmd only
        y_pred, mmd = y_pred

        rmse = nn.MSELoss()
        return rmse(y_pred, y) + self.lam_mmd * mmd


class DR_CRN(nn.Module):
    def __init__(
        self,
        single_cause_index,
        n_confounder,
        n_cause,
        n_outcome,
        n_confounder_rep,
        n_outcome_rep,
        mmd_sigma,
        lam_factual,
        lam_propensity,
        lam_mmd,
        linear=False,
        binary_outcome=False,
        device=DEVICE,
    ):
        super().__init__()
        self.single_cause_index = single_cause_index
        self.n_confounder = n_confounder
        self.n_cause = n_cause
        self.n_outcome = n_outcome
        self.mmd_sigma = mmd_sigma
        self.lam_factual = lam_factual
        self.lam_propensity = lam_propensity
        self.binary_outcome = binary_outcome
        self.lam_mmd = lam_mmd

        n_input = n_confounder + n_cause - 1
        self.n_x = n_confounder + n_cause

        # two representation networks
        if linear:
            self.confounder_rep_net = nn.Sequential(nn.Linear(n_input, n_confounder_rep),).to(device)

            self.outcome_rep_net = nn.Sequential(nn.Linear(n_input, n_outcome_rep),).to(device)
        else:
            self.confounder_rep_net = nn.Sequential(
                nn.Linear(n_input, n_input + 1), nn.ReLU(), nn.Linear(n_input + 1, n_confounder_rep),
            ).to(device)

            self.outcome_rep_net = nn.Sequential(
                nn.Linear(n_input, n_input + 1), nn.ReLU(), nn.Linear(n_input + 1, n_outcome_rep),
            ).to(device)

        # propensity network

        if linear:
            self.propensity_net = nn.Sequential(nn.Linear(n_confounder_rep, 2), nn.LogSoftmax(dim=-1)).to(device)
        else:
            self.propensity_net = nn.Sequential(
                nn.Linear(n_confounder_rep, n_confounder_rep + 1),
                nn.ReLU(),
                nn.Linear(n_confounder_rep + 1, 2),
                nn.LogSoftmax(dim=-1),
            ).to(device)

        # outcome regression network

        if not self.binary_outcome:
            if linear:
                self.outcome_net0 = nn.Sequential(nn.Linear(n_confounder_rep + n_outcome_rep, n_outcome),).to(device)

                self.outcome_net1 = nn.Sequential(nn.Linear(n_confounder_rep + n_outcome_rep, n_outcome),).to(device)
            else:
                self.outcome_net0 = nn.Sequential(
                    nn.Linear(n_confounder_rep + n_outcome_rep, n_confounder_rep + n_outcome_rep + 1),
                    nn.ReLU(),
                    nn.Linear(n_confounder_rep + n_outcome_rep + 1, n_outcome),
                ).to(device)

                self.outcome_net1 = nn.Sequential(
                    nn.Linear(n_confounder_rep + n_outcome_rep, n_confounder_rep + n_outcome_rep + 1),
                    nn.ReLU(),
                    nn.Linear(n_confounder_rep + n_outcome_rep + 1, n_outcome),
                ).to(device)
        else:
            if linear:
                pass
            else:
                # probability
                self.outcome_net0 = nn.Sequential(
                    nn.Linear(n_confounder_rep + n_outcome_rep, n_outcome), nn.Sigmoid()
                ).to(device)

                self.outcome_net1 = nn.Sequential(
                    nn.Linear(n_confounder_rep + n_outcome_rep, n_outcome), nn.Sigmoid()
                ).to(device)

    def forward(self, x):  # pylint: disable=arguments-differ
        # slice single cause

        single_cause = x[:, self.single_cause_index : (self.single_cause_index + 1)]
        # print('single_cause', single_cause.shape)

        if self.single_cause_index == 0:
            all_confounders = x[:, 1:]
        elif self.single_cause_index == self.n_x - 1 or self.single_cause_index == -1:
            all_confounders = x[:, :-1]
        else:
            all_confounders = torch.cat(
                [x[:, : self.single_cause_index], x[:, (self.single_cause_index + 1) :]], dim=-1
            )

        assert all_confounders.shape[1] == self.n_x - 1

        # get representations
        confounder_rep = self.confounder_rep_net(all_confounders)
        outcome_rep = self.outcome_rep_net(all_confounders)

        # print('confounder_rep', confounder_rep.shape)
        # print('outcome_rep', outcome_rep.shape)

        combined_rep = torch.cat([confounder_rep, outcome_rep], dim=-1)

        treated_mask = single_cause == 1.0
        treated_label = treated_mask.to(torch.long)

        # treated_mask = torch.cat([treated_mask] * confounder_rep.shape[1], dim=1)
        rep_treated = confounder_rep[treated_mask.squeeze()]
        rep_control = confounder_rep[~treated_mask.squeeze()]

        log_propensity = self.propensity_net(confounder_rep)

        outcome1 = self.outcome_net1(combined_rep)
        outcome0 = self.outcome_net0(combined_rep)

        outcome = outcome1 * single_cause + outcome0 * (1 - single_cause)

        mmd = mmd2_rbf(rep_treated, rep_control, self.mmd_sigma)
        return outcome, log_propensity, mmd, treated_label

    def loss(self, y_pred, y):
        # y_pred is the output of forward
        # print('y_pred', len(y_pred))
        y_hat, log_propensity, mmd, treated_label = y_pred

        # factual loss
        if not self.binary_outcome:
            rmse = nn.MSELoss()
            # print('y_hat', y_hat.shape)
            # print('y', y.shape)
            error = torch.sqrt(rmse(y_hat, y))
        else:
            neg_y_hat = 1.0 - y_hat
            # N, 2, D_out
            y_hat_2d = torch.cat([y_hat[:, None, :], neg_y_hat[:, None, :]], dim=1)
            y_hat_2d = torch.log(y_hat_2d + 1e-9)
            outcome_nll_loss = nn.NLLLoss()
            # N, D_out
            y = y.to(torch.long)
            error = outcome_nll_loss(y_hat_2d, y)

        # propensity loss
        nll_loss = nn.NLLLoss()
        nll = nll_loss(log_propensity, treated_label.squeeze())

        loss = error * self.lam_factual + nll * self.lam_propensity + mmd * self.lam_mmd

        # print('error', error.item())
        # print('nll', nll.item())
        # print('mmd', mmd.item())

        return loss


class DR_CRN_Multicause(nn.Module):
    def __init__(
        self,
        n_confounder,
        n_cause,
        n_outcome,
        n_confounder_rep,
        n_outcome_rep,
        mmd_sigma=0.0,
        lam_mmd=0.0,
        device=DEVICE,
    ):
        super().__init__()

        assert n_outcome == 1
        self.n_treatment = int(np.power(2, n_cause))
        self.mmd_sigma = mmd_sigma
        self.lam_mmd = lam_mmd
        n_input = n_confounder

        self.confounder_rep_net = nn.Sequential(
            nn.Linear(n_input, n_input + 1), nn.ReLU(), nn.Linear(n_input + 1, n_confounder_rep),
        ).to(device)

        self.outcome_rep_net = nn.Sequential(
            nn.Linear(n_input, n_input + 1), nn.ReLU(), nn.Linear(n_input + 1, n_outcome_rep),
        ).to(device)

        self.propensity_net = nn.Sequential(
            nn.Linear(n_confounder_rep, n_confounder_rep + 1),
            nn.ReLU(),
            nn.Linear(n_confounder_rep + 1, self.n_treatment),
            nn.LogSoftmax(dim=-1),
        ).to(device)

        heads = []

        for i in range(self.n_treatment):
            heads.append(
                nn.Sequential(
                    # nn.Linear(n_outcome_rep + n_confounder_rep, n_outcome_rep + n_confounder_rep + 1),
                    # nn.ReLU(),
                    nn.Linear(n_outcome_rep + n_confounder_rep, 1)
                ).to(device)
            )

        self.heads = nn.ModuleList(heads)

    def forward(self, input_mat):  # pylint: disable=arguments-differ
        # confounder, cause indicator
        x = input_mat[:, :-1]
        cause = input_mat[:, -1].unsqueeze(-1).to(torch.long)
        cause_mat = torch.zeros(x.shape[0], self.n_treatment).to(x)
        cause_mat.scatter_(1, cause, 1)

        # get representations
        confounder_rep = self.confounder_rep_net(x)
        outcome_rep = self.outcome_rep_net(x)
        combined_rep = torch.cat([confounder_rep, outcome_rep], dim=-1)

        predictions = []
        for i in range(self.n_treatment):
            y_hat = self.heads[i](combined_rep)
            predictions.append(y_hat)

        prediction = torch.cat(predictions, dim=-1)
        prediction_selected = torch.sum(prediction * cause_mat, dim=1).unsqueeze(-1)

        log_propensity = self.propensity_net(confounder_rep)
        mmd = 0

        if self.lam_mmd != 0.0:
            # calculate mmd
            cause_np = cause.squeeze().cpu().numpy()
            uniques, counts = np.unique(cause_np, return_counts=True)
            uniques = uniques[counts > 1]

            if len(uniques) < 2:
                print("uniques", uniques)
                print("counts", counts)
                return prediction_selected, 0

            for i in range(len(uniques)):
                for j in range(i + 1, len(uniques)):
                    x1 = confounder_rep[cause.squeeze() == uniques[i]]
                    x2 = confounder_rep[cause.squeeze() == uniques[j]]
                    mmd = mmd + mmd2_rbf(x1, x2, self.mmd_sigma)

        return prediction_selected, log_propensity, mmd, cause

    def loss(self, y_pred, y):
        # y_pred is the output of forward
        y_hat, log_propensity, mmd, treated_label = y_pred

        # factual loss
        rmse = nn.MSELoss()
        error = torch.sqrt(rmse(y_hat, y))

        # propensity loss
        nll_loss = nn.NLLLoss()
        nll = nll_loss(log_propensity, treated_label.squeeze())

        loss = error + nll + mmd * self.lam_mmd

        return loss
