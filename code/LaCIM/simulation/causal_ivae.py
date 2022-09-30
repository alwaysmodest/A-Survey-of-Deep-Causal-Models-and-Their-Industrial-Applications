#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:31:31 2020

@author:  xinsun
"""
import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass
    
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        # pass
        # print('call such')
        nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)
        
        
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.5, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        self.activation = [activation] * (self.n_layers - 1)
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'relu':
                self._act_f.append(F.relu)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h



    
class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):

        eps = self._dist.sample(mu.size()).squeeze()
        # print(mu, v, eps)
        # print(self.device)
        # eps = eps.to(self.device)
        # print(eps,v.sqrt())
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v):
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        return lpdf.sum(dim=-1)

class Causal_VAE(nn.Module):
    def __init__(self, dim_z, dim_s, dim_x, dim_y, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.5, device='cpu', anneal=False):
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_s = dim_s
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.prior_dist = Normal(device=device)
        self.decoder_dist = Normal(device=device)
        self.encoder_dist = Normal(device=device)
        self.device = device
        # prior_params
        self.prior_mean = torch.zeros(dim_z+dim_s).to(device)
        self.logl = torch.zeros(dim_z+dim_s).to(device)
        # decoder x params
        self.fx = MLP(dim_z+dim_s, dim_x, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_x_var = MLP(dim_z+dim_s, dim_x, hidden_dim, n_layers, activation=activation, slope=slope)
        # decode y params
        self.fy = MLP(dim_s, dim_y, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_y_var = MLP(dim_s, dim_y, hidden_dim, n_layers, activation=activation, slope=slope)
        # encoder params
        self.g = MLP(dim_x+dim_y, dim_z+dim_s, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(dim_x+dim_y, dim_z+dim_s, hidden_dim, n_layers, activation=activation, slope=slope)

        self.apply(weights_init)
        self.to(self.device)


    def encoder_params(self, x, y):
        xyc = torch.cat((x, y), 1)
        g = self.g(xyc)
        logv = self.logv(xyc)
        return g, logv.exp()

    def decoder_x_params(self, z,s):
        zs = torch.cat((z,s),1)
        fx = self.fx(zs)
        logx_sigma = self.decoder_x_var(zs)
        return fx, logx_sigma.exp()
    
    def decoder_y_params(self, s):
        fy = self.fy(s)
        logy_sigma = self.decoder_y_var(s)
        return fy, logy_sigma.exp()
    
    def prior_params(self):
        return self.prior_mean, self.logl.exp()

    def forward(self, x, y):
        prior_params = self.prior_params()
        encoder_params = self.encoder_params(x, y)
        zs = self.encoder_dist.sample(*encoder_params)
        z = zs[:,:int(encoder_params[0].shape[1]/2)]
        s = zs[:,int(encoder_params[0].shape[1]/2):]
        decoder_x_params = self.decoder_x_params(z,s)
        decoder_y_params = self.decoder_y_params(s)
        return decoder_x_params, decoder_y_params, encoder_params, z, s, prior_params

    def elbo(self, x, y):
        decoder_x_params, decoder_y_params, (g, v), z,s, prior_params = self.forward(x,y)
        zs = torch.cat((z,s),1)
        log_px_zs = self.decoder_dist.log_pdf(x, *decoder_x_params)
        log_py_s = self.decoder_dist.log_pdf(y, *decoder_y_params)
        log_qzs_xy = self.encoder_dist.log_pdf(zs, g, v)
        # print(zs, *prior_params)
        log_pzs = self.prior_dist.log_pdf(zs, *prior_params)

        return (-log_px_zs - log_py_s + log_qzs_xy - log_pzs).mean(), (-log_px_zs).mean(), - log_py_s.mean(),(log_qzs_xy - log_pzs).mean(),z, s, decoder_x_params,decoder_y_params, prior_params



    
    
class Causal_iVAE(nn.Module):
    def __init__(self, dim_z, dim_s, dim_x, dim_y, dim_c, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.5, device='cpu', anneal=False):
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.prior_dist = Normal(device=device)
        self.decoder_dist = Normal(device=device)
        self.encoder_dist = Normal(device=device)

        # prior_params
        self.prior_mean = MLP(dim_c, dim_s+dim_z, hidden_dim, n_layers, activation=activation, slope=slope)
        # print('dim_c', dim_c,'self.prior_mean',self.prior_mean)
        self.logl = MLP(dim_c, dim_s+dim_z, hidden_dim, n_layers, activation=activation, slope=slope)
        # decoder x params
        self.fx = MLP(dim_z+dim_s, dim_x, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_x_var = MLP(dim_z+dim_s, dim_x, hidden_dim, n_layers, activation=activation, slope=slope)
        # decode y params
        self.fy = MLP(dim_s, dim_y, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_y_var = MLP(dim_s, dim_y, hidden_dim, n_layers, activation=activation, slope=slope)
        # encoder params
        self.g = MLP(dim_x+dim_y+dim_c, dim_z+dim_s, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(dim_x+dim_y+dim_c, dim_z+dim_s, hidden_dim, n_layers, activation=activation, slope=slope)

        self.apply(weights_init)
        self.to(device)


    def encoder_params(self, x, y, c):
        xyc = torch.cat((x, y, c), 1)
        g = self.g(xyc)
        logv = self.logv(xyc)
        return g, logv.exp()

    def decoder_x_params(self, z,s):
        zs = torch.cat((z,s),1)
        fx = self.fx(zs)
        logx_sigma = self.decoder_x_var(zs)
        return fx, logx_sigma.exp()
    
    def decoder_y_params(self, s):
        fy = self.fy(s)
        logy_sigma = self.decoder_y_var(s)
        return fy, logy_sigma.exp()
    
    def prior_params(self, c):
        mean = self.prior_mean(c)
        logl = self.logl(c)
        return mean, logl.exp()

    def forward(self, x, y, c):
        prior_params = self.prior_params(c)
        encoder_params = self.encoder_params(x, y, c)
        zs = self.encoder_dist.sample(*encoder_params)
        z = zs[:,:int(encoder_params[0].shape[1]/2)]
        s = zs[:,int(encoder_params[0].shape[1]/2):]
        decoder_x_params = self.decoder_x_params(z,s)
        decoder_y_params = self.decoder_y_params(s)
        return decoder_x_params, decoder_y_params, encoder_params, z, s, prior_params

    def elbo(self, x, y, c):
        log_px_zs = 0
        log_py_s = 0
        log_qzs_xyc = 0
        log_pzs_c = 0
        for i in range(10):
            decoder_x_params, decoder_y_params, (g, v), z,s, prior_params = self.forward(x,y,c)
            zs = torch.cat((z,s),1)
            log_px_zs += self.decoder_dist.log_pdf(x, *decoder_x_params)
            log_py_s += self.decoder_dist.log_pdf(y, *decoder_y_params)
            log_qzs_xyc += self.encoder_dist.log_pdf(zs, g, v)
            log_pzs_c += self.prior_dist.log_pdf(zs, *prior_params)
        log_px_zs = log_px_zs / 10
        log_py_s = log_py_s / 10
        log_qzs_xyc = log_qzs_xyc / 10
        log_pzs_c = log_pzs_c / 10
        return (-log_px_zs - log_py_s + log_qzs_xyc - log_pzs_c).mean(), (-log_px_zs).mean(), - log_py_s.mean(),(log_qzs_xyc - log_pzs_c).mean(),z, s, decoder_x_params,decoder_y_params, prior_params

    
