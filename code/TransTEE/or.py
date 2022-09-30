import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
import numpy as np
import pandas as pd

from data.datasets import twin_datasets
from torch.utils.data import DataLoader
from models.subspaceTransformerITE import AE, OutNet, Discriminator, SubTransCasual, TransformerEncoder
import argparse
import random
import itertools
from utils.metrics import PEHE, ATE

from scipy.spatial.distance import pdist, squareform
import scipy

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

# 随机数种子
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)


def mainSubITE(args):
    # 构造数据集
    train_dataset = twin_datasets(isTrain=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # T=0 representation
    encoder_t0 = SubTransCasual(args.input_dims, args.hid_dims, 1, 1, 3, 3).to(device)

    # T=1 representation
    encoder_t1 = SubTransCasual(args.input_dims, args.hid_dims, 1, 1, 3, 3).to(device)

    # common representation
    encoder_common = SubTransCasual(args.input_dims, args.hid_dims, 1, 1, 3, 3).to(device)

    # Encoder-Decoder
    en_decoder = AE(args.input_dims * 2, args.hid_dims, args.input_dims).to(device)

    # Discriminator
    discriminator = Discriminator(args.input_dims, args.hid_dims, 1).to(device)

    # outNet
    outnet_t0 = OutNet(args.hid_dims * 2, 1).to(device)
    outnet_t1 = OutNet(args.hid_dims * 2, 1).to(device)

    # optimizer
    # optimizer_G = Adam(itertools.chain(encoder_t0.parameters(), encoder_t1.parameters(), encoder_common.parameters(),
    #                                    outnet_t0.parameters(), outnet_t1.parameters(), en_decoder.parameters()), lr=args.lr)
    optimizer_G = Adam(itertools.chain(encoder_t0.parameters(), encoder_t1.parameters(), encoder_common.parameters(),
                                       outnet_t0.parameters(), outnet_t1.parameters()),lr=args.lr)
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr)


    # results
    min_pehe = 99999
    min_ate = 99999


    # train
    for epoch in range(args.epoch):
        for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
            train_x = train_x.float().to(device)
            train_t = train_t.float().to(device)
            train_y = train_y.float().to(device)

            input_x0 = train_x[train_t == 0].float().to(device)
            input_y0 = train_y[train_t == 0].float().to(device)

            input_x1 = train_x[train_t == 1].float().to(device)
            input_y1 = train_y[train_t == 1].float().to(device)

            # three embeddings(t=0, t=1, t=common)
            embedding_t0, pseudo_predict_t0 = encoder_t0(input_x0)
            embedding_t1, pseudo_predict_t1 = encoder_t1(input_x1)
            embedding_common, _ = encoder_common(train_x)

            # squeeze()
            embedding_t0, pseudo_predict_t0, embedding_t1, pseudo_predict_t1, embedding_common = \
                embedding_t0.squeeze(), pseudo_predict_t0.squeeze(), embedding_t1.squeeze(), pseudo_predict_t1.squeeze(), embedding_common.squeeze()

            embedding_common_t0 = torch.cat([embedding_t0, embedding_common[train_t == 0]], dim=-1)
            embedding_common_t1 = torch.cat([embedding_t1, embedding_common[train_t == 1]], dim=-1)

            # reconstruction results
            input_x0_hat = en_decoder(embedding_common_t0)
            input_x1_hat = en_decoder(embedding_common_t1)

            # prediction results
            y_hat_0 = outnet_t0(embedding_common_t0)
            y_hat_1 = outnet_t1(embedding_common_t1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for i in range(args.dis_epoch):
                optimizer_D.zero_grad()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(embedding_common[train_t == 0])) + torch.mean(discriminator(embedding_common[train_t == 1]))
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

            # ---------------------
            #  Train others
            # ---------------------
            # loss_G
            loss_G = -torch.mean(discriminator(embedding_common[train_t == 1]))

            # loss pseudo t0 t1
            loss_pseudo = F.mse_loss(pseudo_predict_t0, input_y0).to(device) + F.mse_loss(pseudo_predict_t1, input_y1).to(device)

            # loss reconstruction
            loss_recons = F.mse_loss(input_x0_hat, input_x0).to(device) + F.mse_loss(input_x1_hat, input_x1).to(device)

            # loss prediction
            loss_pred = F.l1_loss(y_hat_0.squeeze(), input_y0).to(device) + F.l1_loss(y_hat_1.squeeze(), input_y1).to(device)

            # total loss
            loss_other = loss_G + loss_pseudo + loss_recons + 10*loss_pred

            optimizer_G.zero_grad()
            loss_other.backward()
            optimizer_G.step()

            if steps % args.print_steps == 0 or steps == 0:
                print(
                    "Epoches: %d, step: %d, loss_D:%.3f, loss_G:%.3f, loss_pseudo:%.3f, loss_recons:%.3f, loss_pred:%.3f"
                    % (epoch, steps, loss_D.detach().cpu().numpy(), loss_G.detach().cpu().numpy(), loss_pseudo.detach().cpu().numpy(), loss_recons.detach().cpu().numpy(), loss_pred.detach().cpu().numpy()))
                # ---------------------
                #         Test
                # ---------------------
                encoder_t0.eval()
                encoder_t1.eval()
                encoder_common.eval()
                en_decoder.eval()
                discriminator.eval()
                outnet_t0.eval()
                outnet_t1.eval()

                total_test_potential_y = torch.Tensor([]).to(device)
                total_test_potential_y_hat = torch.Tensor([]).to(device)
                test_dataset = twin_datasets(isTrain=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
                for steps, [test_x, test_potential_y] in enumerate(test_dataloader):
                    test_x = test_x.float().to(device)
                    test_potential_y = test_potential_y.float().to(device)
                    embedding_t0, _ = encoder_t0(test_x)
                    embedding_t1, _ = encoder_t1(test_x)
                    embedding_common, _ = encoder_common(test_x)
                    embedding_common_t0 = torch.cat([embedding_t0, embedding_common], dim=-1)
                    embedding_common_t1 = torch.cat([embedding_t1, embedding_common], dim=-1)
                    y_hat_0 = outnet_t0(embedding_common_t0).squeeze(2)
                    y_hat_1 = outnet_t1(embedding_common_t1).squeeze(2)

                    test_potential_y_hat = torch.cat([y_hat_0, y_hat_1], dim=-1)

                    total_test_potential_y = torch.cat([total_test_potential_y, test_potential_y], dim=0)
                    total_test_potential_y_hat = torch.cat([total_test_potential_y_hat, test_potential_y_hat], dim=0)

                pehe = PEHE(total_test_potential_y_hat.cpu().detach().numpy(), total_test_potential_y.cpu().detach().numpy())
                ate = ATE(total_test_potential_y_hat.cpu().detach().numpy(), total_test_potential_y.cpu().detach().numpy())

                print("PEHE:", pehe)
                print("ATE:", ate)
                min_pehe = min(pehe, min_pehe)
                min_ate = min(ate, min_ate)
                encoder_t0.train()
                encoder_t1.train()
                encoder_common.train()
                en_decoder.train()
                discriminator.train()
                outnet_t0.train()
                outnet_t1.train()
    print("PEHE:", min_pehe)
    print("ATE:", min_ate)

    # torch.save(autoencoder, 'checkpoint/autoencode.pth')
    # torch.save(outNet, 'checkpoint/outNet.pth')
    # print("模型保存完毕")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--input_dims", default=30, type=int)
    parser.add_argument("--hid_dims", default=30, type=int)
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--print_steps", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=int)
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--dis_epoch", type=int, default=2, help="discrimator epoch")

    args = parser.parse_args()
    print(args)

    if (torch.cuda.is_available()):
        print("GPU is ready \n")
    else:
        print("CPU is ready \n")

    # main(args)
    mainSubITE(args)
