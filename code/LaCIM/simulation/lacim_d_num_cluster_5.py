# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:26:49 2020

@author: xinsun
"""

import os
import pickle
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from metrics import mean_corr_coef
# from generate_discrete_c import SyntheticDataset
# from generate_discrete_c import save_data
from generate_continuous_c import SyntheticDataset
from generate_continuous_c import save_data
from causal_ivae import Causal_iVAE
from causal_ivae import Causal_VAE

dim_z = 2
dim_s = 2
dim_x = 4
dim_y = 2
dim_c = 2
n_per_clus = 1000
num_cluster = 5
n_layers = 3
activation = 'lrelu'

# IVAE

num_data = 5
num_trial = 20
z_perf_ivae_discrete_5 = np.zeros([num_data, num_trial])
s_perf_ivae_discrete_5 = np.zeros([num_data, num_trial])
z_ivae_list_data_discrete_5 = []
s_ivae_list_data_discrete_5 = []
hist_ivae_perf_z_data_discrete_5 = []
hist_ivae_perf_s_data_discrete_5 = []
i = 0
print('train 5 times')
while i < 5:
    hist_ivae_perf_z_discrete_5 = []
    hist_ivae_perf_s_discrete_5 = []
    dim_c = 2
    path = save_data(num_cluster, n_per_clus, dim_c, dim_z, dim_s, dim_x, dim_y, n_layers, activation, slope=0.5,
                     dtype=np.float32, seed=i)
    path += '.npz'
    print(path)
    dset = SyntheticDataset(path, 'cpu')
    N = dset.len
    train_loader = DataLoader(dset, shuffle=True, batch_size=512)
    x_gt = dset.x
    y_gt = dset.y
    c_gt = dset.c
    z_gt = dset.z
    s_gt = dset.s
    label_gt = dset.label
    dim_c = label_gt.shape[1]
    j = 0
    while j < 20:
        model_ivae = Causal_iVAE(dim_z, dim_s, dim_x, dim_y, dim_c)
        optimizer_ivae = optim.Adam(model_ivae.parameters(), lr=0.5 * 1e-3)
        scheduler_ivae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ivae, factor=0.1, patience=10, verbose=True)
        hist_ivae_perf_z = []
        hist_ivae_perf_s = []
        z_ivae_list = []
        s_ivae_list = []
        start_time = time.time()
        for k in range(4000):
            if k % 400 == 0 and k > 0:
                print(i, j, k, 'using: ', time.time() - start_time)
                start_time = time.time()
            x, y, s, z, _, c = next(iter(train_loader))
            optimizer_ivae.zero_grad()
            elbo, elbo_1, elbo_2, elbo_5, z, s, decoder_x_params, decoder_y_params, prior_params = model_ivae.elbo(x, y,
                                                                                                                   c)
            elbo.backward()
            optimizer_ivae.step()
            if torch.isnan(elbo) or torch.isinf(elbo):
                break
            
            _, _, _, _, z_ivae, s_ivae, _, _, _ = model_ivae.elbo(x_gt, y_gt, label_gt)
            if torch.sum(torch.isnan(z_ivae)) == 0 and torch.sum(torch.isnan(s_ivae)) == 0:
                perf_z = mean_corr_coef(z_gt.numpy(), z_ivae.detach().numpy())
                perf_s = mean_corr_coef(s_gt.numpy(), s_ivae.detach().numpy())
                hist_ivae_perf_z.append(perf_z)
                hist_ivae_perf_s.append(perf_s)
        
        if torch.isnan(elbo) or torch.isinf(elbo):
            continue
        
        label = dset.label
        _, _, _, _, z_ivae, s_ivae, _, _, _ = model_ivae.elbo(x_gt, y_gt, label_gt)
        perf_ivae_z = mean_corr_coef(z_gt.numpy(), z_ivae.detach().numpy())
        perf_ivae_s = mean_corr_coef(s_gt.numpy(), s_ivae.detach().numpy())
        z_ivae_list.append(z_ivae)
        s_ivae_list.append(s_ivae)
        z_perf_ivae_discrete_5[i, j] = perf_ivae_z
        s_perf_ivae_discrete_5[i, j] = perf_ivae_s
        hist_ivae_perf_z_discrete_5.append(hist_ivae_perf_z)
        hist_ivae_perf_s_discrete_5.append(hist_ivae_perf_s)
        j += 1
        print('This is the (%s,%s)-th experiment, the z_perf is %0.5f, the s_perf is %0.5f' % (
        i, j, perf_ivae_z, perf_ivae_s))
        
        if not os.path.exists(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/'):
            os.makedirs(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/')
        with open(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/check.pkl', 'wb') as f:
            dicts = {
                'i': i, 'j': j, 'k': k,
                'z_perf_ivae_discrete_5': z_perf_ivae_discrete_5,
                's_perf_ivae_discrete_5': s_perf_ivae_discrete_5,
                'z_ivae_list_data_discrete_5': z_ivae_list_data_discrete_5,
                's_ivae_list_data_discrete_5': s_ivae_list_data_discrete_5,
                'hist_ivae_perf_z_data_discrete_5': hist_ivae_perf_z_data_discrete_5,
                'hist_ivae_perf_s_data_discrete_5': hist_ivae_perf_s_data_discrete_5
            }
            pickle.dump(dicts, f)
    
    z_ivae_list_data_discrete_5.append(z_ivae_list)
    s_ivae_list_data_discrete_5.append(s_ivae_list)
    hist_ivae_perf_z_data_discrete_5.append(hist_ivae_perf_z_discrete_5)
    hist_ivae_perf_s_data_discrete_5.append(hist_ivae_perf_s_discrete_5)
    i += 1
    
    with open(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/check.pkl', 'wb') as f:
        dicts = {
            'i': i, 'j': j, 'k': k,
            'z_perf_ivae_discrete_5': z_perf_ivae_discrete_5,
            's_perf_ivae_discrete_5': s_perf_ivae_discrete_5,
            'z_ivae_list_data_discrete_5': z_ivae_list_data_discrete_5,
            's_ivae_list_data_discrete_5': s_ivae_list_data_discrete_5,
            'hist_ivae_perf_z_data_discrete_5': hist_ivae_perf_z_data_discrete_5,
            'hist_ivae_perf_s_data_discrete_5': hist_ivae_perf_s_data_discrete_5
        }
        pickle.dump(dicts, f)
    
    with open(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/final.pkl', 'wb') as f:
        dicts = {
            'i': i, 'j': j, 'k': k,
            'z_perf_ivae_discrete_5': z_perf_ivae_discrete_5,
            's_perf_ivae_discrete_5': s_perf_ivae_discrete_5,
            'z_ivae_list_data_discrete_5': z_ivae_list_data_discrete_5,
            's_ivae_list_data_discrete_5': s_ivae_list_data_discrete_5,
            'hist_ivae_perf_z_data_discrete_5': hist_ivae_perf_z_data_discrete_5,
            'hist_ivae_perf_s_data_discrete_5': hist_ivae_perf_s_data_discrete_5
        }
        pickle.dump(dicts, f)

