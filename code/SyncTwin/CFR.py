import torch
import torch.nn as nn

from config import D_TYPE, DEVICE


class CFR(nn.Module):
    def __init__(self, n_unit, n_treated, lam_dist=1.0, encoder=None, decoder_Y=None, device=DEVICE, dtype=D_TYPE):
        super(CFR, self).__init__()

        self.n_unit = n_unit
        self.n_treated = n_treated
        self.encoder = encoder.to(device)
        if decoder_Y is not None:
            self.decoder_Y = decoder_Y.to(device)
        self.device = device

        # regularization strength of distributional distance
        self.lam_dist = lam_dist
        self.p = 0.5
        self.sig = 1.0

    def check_device(self, *args):
        a_list = []
        for a in args:
            if a.device != self.device:
                res = a.to(self.device)
            else:
                res = a
            a_list.append(res)
        return a_list

    def get_representation(self, x, t, mask):
        # get representation C: B(atch size), D(im hidden)
        x, t, mask = self.check_device(x, t, mask)  # pylint: disable=unbalanced-tuple-unpacking
        C = self.encoder(x, t, mask)
        return C

    def get_prognostics(self, C, t, mask):
        C, t, mask = self.check_device(C, t, mask)  # pylint: disable=unbalanced-tuple-unpacking
        y_hat = self.decoder_Y(C, t, mask)
        return y_hat

    def prognostic_loss(self, y, y_hat, mask=None):
        if mask is None:
            mask = torch.ones_like(y)
        y, y_hat, mask = self.check_device(y, y_hat, mask)  # pylint: disable=unbalanced-tuple-unpacking
        err = (y - y_hat) * mask
        err_mse = torch.sum(err ** 2) / torch.sum(mask)
        return err_mse

    def forward(self, x, t, mask, y, y_mask):
        x, t, mask, y, y_mask = self.check_device(x, t, mask, y, y_mask)  # pylint: disable=unbalanced-tuple-unpacking
        C = self.get_representation(x, t, mask)
        y_hat = self.get_prognostics(C, t, y_mask)

        y_control = y[:, : self.n_unit, :]

        p_loss = self.prognostic_loss(y_control, y_hat[:, : self.n_unit, :])

        if self.lam_dist != 0:
            C_c = C[: self.n_unit, :]
            C_t = C[self.n_unit :, :]
            mmd = mmd2_rbf(C_c, C_t, self.p, self.sig)
            p_loss = p_loss + mmd

        return p_loss

    def get_treatment_effect(self, x, t, mask, y, y_mask):
        x, t, mask, y, y_mask = self.check_device(x, t, mask, y, y_mask)  # pylint: disable=unbalanced-tuple-unpacking
        C = self.get_representation(x, t, mask)
        y_hat = self.get_prognostics(C, t, y_mask)
        y_hat = y_hat[:, self.n_unit :, :]
        y = y[:, self.n_unit :, :]

        te = y - y_hat
        return te


def mmd2_rbf(Xc, Xt, p, sig):
    """ Computes the l2-RBF MMD for X given t """

    Kcc = torch.exp(-pdist2sq(Xc, Xc) / (sig ** 2))
    Kct = torch.exp(-pdist2sq(Xc, Xt) / (sig ** 2))
    Ktt = torch.exp(-pdist2sq(Xt, Xt) / (sig ** 2))

    m = Xc.shape[0] * 1.0
    n = Xt.shape[0] * 1.0

    mmd = ((1.0 - p) ** 2) / (m * (m - 1.0)) * (torch.sum(Kcc) - m)
    mmd = mmd + (p ** 2) / (n * (n - 1.0)) * (torch.sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * torch.sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * torch.matmul(X, Y.T)
    nx = torch.sum(X ** 2, dim=1).unsqueeze(1)
    ny = torch.sum(Y ** 2, dim=1).unsqueeze(1)
    D = (C + ny.T) + nx
    return D
