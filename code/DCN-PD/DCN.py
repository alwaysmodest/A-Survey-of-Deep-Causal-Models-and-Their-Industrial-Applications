import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from Utils import Utils


class DCN(nn.Module):
    def __init__(self, training_flag):
        super(DCN, self).__init__()
        self.training = training_flag

        # shared layer
        self.shared1 = nn.Linear(in_features=25, out_features=200)
        nn.init.xavier_uniform_(self.shared1.weight)

        self.shared2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.shared2.weight)

        # potential outcome1 Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y1.weight)

        self.hidden2_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y1.weight)

        self.out_Y1 = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.out_Y1.weight)

        # potential outcome1 Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y0.weight)

        self.hidden2_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y0.weight)

        self.out_Y0 = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.out_Y0.weight)

    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training:
            y1, y0 = self.__train_net(x, ps_score)
        else:
            y1, y0 = self.__eval_net(x)

        return y1, y0

    def __train_net(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # shared layers
        shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))
        x = F.relu(shared_mask * self.shared1(x))
        x = F.relu(shared_mask * self.shared2(x))

        # potential outcome1 Y(1)
        y1_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0

    def __eval_net(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0
