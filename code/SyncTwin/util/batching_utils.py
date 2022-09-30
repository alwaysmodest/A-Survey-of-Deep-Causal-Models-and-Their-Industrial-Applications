import numpy as np
import numpy.random
import torch

from config import DEVICE


def get_batch_all(batch_size, n_units_total, n_units, x, y, y_control, device=DEVICE):
    batch_ind = numpy.random.choice(np.arange(n_units_total), batch_size)
    batch_ind = torch.tensor(batch_ind, dtype=torch.long).to(device)

    x_batch = x[:, batch_ind, :]
    y_batch = y[:, batch_ind, :]

    t_batch = torch.ones_like(x_batch)
    mask_batch = torch.ones_like(x_batch)
    y_mask_batch = (batch_ind < n_units) * 1.0
    return (x_batch, t_batch, mask_batch, batch_ind, y_batch, y_control, y_mask_batch), batch_ind


def get_batch_standard(batch_size, *args):
    array_list = args
    n_total = args[0].shape[1]

    # batching on dim=1
    batch_ind = numpy.random.choice(np.arange(n_total), batch_size)
    batch_ind = torch.tensor(batch_ind, dtype=torch.long).to(args[0].device)

    mini_batch = []

    for a in array_list:
        mini_batch.append(a[:, batch_ind, ...])
    return mini_batch


def get_folds(start, split, *args):
    a = args
    ret = []

    for x in a:
        if x.dim() == 3:
            y = x[:, start::split, :]
        else:
            y = x[start::split]
        ret.append(y)
    return ret


def get_splits(start=0, split=1, args=None):
    # split on first dimension
    a = args
    ret = []

    for x in a:
        y = x[start::split]
        ret.append(y)
    return ret


def get_split_inference(n0_val, n1_val, n_units_val, arr_list, group, nth=0):
    # group = 0, 1
    ret = []
    for a in arr_list:
        if a.dim() == 3:
            if group == 0:
                a_ret = a[:, (nth * n0_val) : (nth * n0_val + n0_val), :]
            else:
                a_ret = a[:, (n_units_val + nth * n1_val) : (n_units_val + nth * n1_val + n1_val), :]
        else:
            if group == 0:
                a_ret = a[(nth * n0_val) : (nth * n0_val + n0_val)]
            else:
                a_ret = a[(n_units_val + nth * n1_val) : (n_units_val + nth * n1_val + n1_val)]
        ret.append(a_ret)
    return ret
