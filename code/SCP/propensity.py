import numpy as np
import torch
import torch.nn as nn

from global_config import DEVICE


def reparameterize(mu, log_sigma):
    std = torch.exp(log_sigma)
    eps = torch.randn_like(std)
    return eps * std + mu


class DensityRatioNetwork(nn.Module):
    def __init__(self, n_confounder, n_z, n_hidden, device=DEVICE):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_confounder + n_z, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 2), nn.LogSoftmax(dim=-1)
        ).to(device)

    def forward(self, x):  # pylint: disable=arguments-differ
        log_prob = self.mlp(x)
        return log_prob

    def loss(self, y_hat, y):
        y = y.to(torch.long).squeeze()
        nll_loss = nn.NLLLoss()
        error = nll_loss(y_hat, y)
        return error


class TreatmentVAE(nn.Module):
    def __init__(self, n_cause, n_hidden, device=DEVICE):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(n_cause, n_hidden * 2), nn.ReLU()).to(device)

        self.fc_mu = nn.Linear(n_hidden * 2, n_hidden).to(device)

        self.fc_log_sigma = nn.Linear(n_hidden * 2, n_hidden).to(device)

        self.decoder = nn.Sequential(nn.Linear(n_hidden, n_cause), nn.Sigmoid()).to(device)

    def encode(self, x):
        latent = self.encoder(x)
        mu = self.fc_mu(latent)
        log_sigma = self.fc_log_sigma(latent)
        return mu, log_sigma

    def decode(self, z):
        prob = self.decoder(z)
        return prob

    def sample_z(self, x):
        mu, log_sigma = self.encode(x)
        z = reparameterize(mu, log_sigma)
        return z

    def forward(self, x):  # pylint: disable=arguments-differ
        mu, log_sigma = self.encode(x)
        z = reparameterize(mu, log_sigma)
        return mu, log_sigma, self.decode(z), x

    def recon_loss(self, prob, y):
        neg_y_hat = 1.0 - prob
        # N, 2, D_out
        y_hat_2d = torch.cat([prob[:, None, :], neg_y_hat[:, None, :]], dim=1)
        y_hat_2d = torch.log(y_hat_2d + 1e-9)
        outcome_nll_loss = nn.NLLLoss()
        # N, D_out
        y = y.to(torch.long)
        error = outcome_nll_loss(y_hat_2d, y)
        return error

    def loss(self, y_pred, y):
        mu, log_sigma, prob, x = y_pred
        log_var = 2.0 * log_sigma
        recon_loss = self.recon_loss(prob, y)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recon_loss + kld_loss
        return loss


class PropensityNetwork(nn.Module):
    def __init__(self, n_confounder, n_cause, n_hidden, device=DEVICE):
        super().__init__()

        self.n_treatment = int(np.power(2, n_cause))
        self.mlp = nn.Sequential(
            nn.Linear(n_confounder, n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            nn.Linear(n_hidden, self.n_treatment),
            nn.LogSoftmax(dim=-1),
        ).to(device)

        self.device = device

    def forward(self, input_mat):  # pylint: disable=arguments-differ
        # input_mat = confounder
        log_propensity = self.mlp(input_mat)
        return log_propensity

    def loss(self, y_pred, y):
        y = y.to(torch.long)
        outcome_nll_loss = nn.NLLLoss()
        # N, D_out
        error = outcome_nll_loss(y_pred, y)
        return error

    def get_weight(self, x, y):
        with torch.no_grad():
            y = y.to(torch.long)
            log_propensity = self.mlp(x)
            nll_loss = nn.NLLLoss(reduction="none")
            ll = nll_loss(log_propensity, y)
            print(torch.mean(ll))
            ll = ll - torch.mean(ll)
            prob = torch.exp(ll)
        return prob

    def get_overlap_weight(self, x, y):
        with torch.no_grad():
            y = y.to(torch.long)
            log_propensity = self.mlp(x)

            denominator = torch.sum(torch.exp(log_propensity * -1), dim=-1)

            nll_loss = nn.NLLLoss(reduction="none")
            ll = nll_loss(log_propensity, y)
            numerator = torch.exp(ll)
            weights = numerator / denominator
            weights = weights / torch.sum(weights) * len(weights)
        return weights
