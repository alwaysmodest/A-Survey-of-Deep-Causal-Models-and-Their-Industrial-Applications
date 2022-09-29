import math

import numpy as np
import sklearn.model_selection as sklearn
import torch
from torch.distributions import Bernoulli


class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def test_train_split(covariates_X, treatment_Y, split_size=0.8):
        return sklearn.train_test_split(covariates_X, treatment_Y, train_size=split_size)

    @staticmethod
    def convert_to_tensor(X, Y):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_DCN(X, ps_score, Y_f, Y_cf):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_ps_score = torch.from_numpy(ps_score)
        tensor_y_f = torch.from_numpy(Y_f)
        tensor_y_cf = torch.from_numpy(Y_cf)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score,
                                                           tensor_y_f, tensor_y_cf)
        return processed_dataset

    @staticmethod
    def concat_np_arr(X, Y, axis=1):
        return np.concatenate((X, Y), axis)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def get_shanon_entropy(prob):
        if prob == 1:
            return -(prob * math.log(prob))
        elif prob == 0:
            return -((1 - prob) * math.log(1 - prob))
        else:
            return -(prob * math.log(prob)) - ((1 - prob) * math.log(1 - prob))

    @staticmethod
    def get_dropout_probability(entropy, gama=1):
        return 1 - (gama * 0.5) - (entropy * 0.5)

    @staticmethod
    def get_dropout_mask(prob, x):
        return Bernoulli(torch.full_like(x, 1 - prob)).sample() / (1 - prob)
