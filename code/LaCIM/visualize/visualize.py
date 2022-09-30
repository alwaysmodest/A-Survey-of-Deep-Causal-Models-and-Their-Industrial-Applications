import os
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import copy
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, type = '3d'):
        super(UnFlatten, self).__init__()
        self.type = type
    def forward(self, input):
        if self.type == '3d':
            return input.view(input.size(0), input.size(1), 1, 1, 1)
        else:
            return input.view(input.size(0), input.size(1), 1, 1)

class Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 decoder_type=0,
                 total_env=2,
                 args=None,
                 is_cuda=1
                 ):
        
        super(Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        self.u_dim = total_env
        print('z_dim is ', self.z_dim)
        self.s_dim = int(self.zs_dim - self.z_dim)
        self.mean_z = []
        self.logvar_z = []
        self.mean_s = []
        self.logvar_s = []
        self.shared_s = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
        for env_idx in range(self.total_env):
            self.mean_z.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.logvar_z.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.mean_s.append(
                nn.Sequential(
                    self.shared_s,
                    nn.Linear(self.in_plane, self.s_dim)
                )
            )
            self.logvar_s.append(
                nn.Sequential(
                    self.shared_s,
                    nn.Linear(self.in_plane, self.s_dim)
                )
            )
        
        self.mean_z = nn.ModuleList(self.mean_z)
        self.logvar_z = nn.ModuleList(self.logvar_z)
        self.mean_s = nn.ModuleList(self.mean_s)
        self.logvar_s = nn.ModuleList(self.logvar_s)
        
        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))
        self.logvar_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))
        
        self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))
    
    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        if env_idx == 0:
                            mu_0 = mu.clone()
                        else:
                            mu_1 = mu.clone()
                            
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = zs[:, :self.z_dim]
                        s = zs[:, self.z_dim:]
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()
                        
                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x).view(-1, 3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]
            
            z, s = z_init, s_init
            z_init_save, s_init_save = z_init.clone(), s_init.clone()
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True
            
            optimizer = optim.Adam(params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)
            
            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x).view(-1, 3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1, 3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y, z_init_save, s_init_save, z, s, mu_0, mu_1
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = zs[:, self.z_dim:]
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            mu_prior, logvar_prior = self.encode_prior(x, env)
            zs = self.reparametrize(mu, logvar)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s
    
    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = zs[:, :self.z_dim]
        s = zs[:, self.z_dim:]
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y
    
    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y
    
    def get_y(self, s):
        return self.Dec_y(s)
    
    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = zs[:, self.z_dim:]
        return self.Dec_y(s)
    
    def encode(self, x, env_idx):
        return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)], dim=1), \
               torch.cat([self.logvar_z[env_idx](x), self.logvar_s[env_idx](x)], dim=1)
    
    def encode_prior(self, x, env_idx):
        temp = env_idx * torch.ones(x.size()[0], 1)
        temp = temp.long().cuda()
        y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, temp, 1)
        # print(env_idx, y_onehot, 'onehot')
        u = self.Enc_u_prior(y_onehot)
        return self.mean_zs_prior(u), self.logvar_zs_prior(u)
    
    def decode_x(self, zs):
        return self.Dec_x(zs)
    
    def decode_y(self, s):
        return self.Dec_y(s)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)
    
    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(128, 128),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(64, 64),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(32, 32),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )
    
    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )
    
    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
    
    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer
    
    def TConv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0,
                      bias=True, groups=1):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                               groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer
    
    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer

class get_dataset_2D_env(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 transform=None):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.transform = transform
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        if args.dataset == 'mnist_2':
            if self.root is None:
                self.root = './data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
            else:
                self.root = self.root + '%s/'%fold
            
            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 4:
                        self.y.append(0)
                    else:
                        self.y.append(1)
        
        print(self.root)
    def __getitem__(self, index):
        #print(self.image_path_list[index])
        with open(self.image_path_list[index], 'rb') as f:
            img_1 = Image.open(f)
            img_1 = Image.fromarray(np.asarray(img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
               torch.from_numpy(np.array(self.y[index]).astype('int')), \
               torch.from_numpy(np.array(self.env[index]).astype('int')), \
               torch.from_numpy(np.array(self.u[index]).astype('float32').reshape((1)))
    
    def __len__(self):
        return len(self.image_path_list)

def evaluate(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    z_init = np.zeros((dataloader.dataset.__len__(), 16))
    s_init = np.zeros((dataloader.dataset.__len__(), 16))
    z = np.zeros((dataloader.dataset.__len__(), 16))
    s = np.zeros((dataloader.dataset.__len__(), 16))
    zs_0 = np.zeros((dataloader.dataset.__len__(), 32))
    zs_1 = np.zeros((dataloader.dataset.__len__(), 32))
    label = np.zeros((dataloader.dataset.__len__(), ))
    x_img = np.zeros((dataloader.dataset.__len__(), 3, 28, 28))
    
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        pred_y_init, pred_y, z_init_t, s_init_t, z_t, s_t, mu_0, mu_1 = model(x, is_train = 0, is_debug=1)
        z_init[batch_begin:batch_begin+x.size(0), :] = z_init_t.detach().cpu().numpy()
        s_init[batch_begin:batch_begin+x.size(0), :] = s_init_t.detach().cpu().numpy()
        z[batch_begin:batch_begin+x.size(0), :] = z_t.detach().cpu().numpy()
        s[batch_begin:batch_begin+x.size(0), :] = s_t.detach().cpu().numpy()
        zs_0[batch_begin:batch_begin+x.size(0), :] = mu_0.detach().cpu().numpy()
        zs_1[batch_begin:batch_begin+x.size(0), :] = mu_1.detach().cpu().numpy()
        pred[batch_begin:batch_begin+x.size(0), :] = pred_y.detach().cpu().numpy()
        label[batch_begin:batch_begin+x.size(0), ] = target.detach().cpu().numpy()
        x_img[batch_begin:batch_begin+x.size(0), :,:,:] = x.detach().cpu().numpy()

        pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()). \
                                              reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
        accuracy_init.update(compute_acc(np.array(pred_y_init.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))

        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    print('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    return pred, label, accuracy.avg, z_init, s_init, z, s, x_img, zs_0, zs_1

def compute_acc(pred, target):
    return (np.sum(np.argmax(pred, axis=1) == target).astype('int')) / pred.shape[0]

class params(object):
    def __init__(self):
        self.in_channel = 3
        self.zs_dim = 32
        self.num_classes = 2
        self.env_num = 2
        self.z_ratio = 0.5
        self.lr2 = 0.007
        self.reg2 = 0.08
        self.test_ep = 100
        self.dataset = 'mnist_2'
        self.data_process = 'fill_std'
        self.eval_path = './results/mnist_2/save_model_VAE_f_share_decoder_test_32_4704.0_0.1_lr_0.0100_80_0.5000_wd_0.0005_2021-03-28_18-25-11/'
        self.root = './data/colored_MNIST_0.02_env_2_0_c_2_2/'
        self.model = 'VAE_f_share'#'shared'#
        self.u_dim = 1
        self.us_dim = 1
        self.is_use_u = 1
        self.test_batch_size = 256
        self.cuda=1
        self.sample_num=10
        self.image_size=28
        self.eval_optim='sgd'
        self.use_best = 1
        self.mse_loss = 1
        self.is_sample = 1

args = params()
test_loader = DataLoaderX(get_dataset_2D_env(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True)
model = Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder(in_channel=args.in_channel,
                                                                 zs_dim=args.zs_dim,
                                                                 num_classes=args.num_classes,
                                                                 decoder_type=1,
                                                                 total_env=args.env_num,
                                                                 args=args
                                                                 ).cuda()

    
check = torch.load('%s/checkpoints.pth.tar' % args.eval_path, map_location=torch.device('cpu')) #best_acc
model.load_state_dict(check['state_dict'], strict=True)
model = model.cuda()
pred, label,_,z_init, s_init, z, s, x_img, mu_0, mu_1 = evaluate(0, model, test_loader, args)

model.eval()
nums = 6
idxs = 10
instance_num = [10]
max_val, min_val = max(z[:, idxs]), min(z[:, idxs])
delta = (max_val - min_val) / (nums-1)
pred, label,_,z_init, s_init, z, s, x_img, mu_0, mu_1 = evaluate(0, model, test_loader, args)
for i in range(z_init.shape[0]):
    if i in instance_num:
        raw_x = x_img[i,:,:,:].transpose((1,2,0)) * 0.5 + 0.5
        img_list = [raw_x]
        for ii in range(6):
            z[i, idxs] = min_val + delta * (ii-1)
            img_list.append(model.decode_x(torch.cat([torch.from_numpy(z[i,:]).unsqueeze(0).cuda().float(), 
                                          torch.from_numpy(s[i,:]).unsqueeze(0).cuda().float()], dim=1))[:, :, 2:30, 2:30].contiguous().squeeze(0).detach().permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
        plot_img = np.concatenate(img_list, axis = 1)
        plt.imshow(plot_img)
        if not os.path.exists('./saved_img_fix_s/'):
            os.makedirs('./saved_img_fix_s/')
        plt.imsave('./saved_img_fix_s/fix_s_%05d.png'%i, plot_img)

# print('begin fix z')
model.eval()
nums = 6
max_val, min_val = 2.5, -0.5
delta = (max_val - min_val) / (nums-1)
pred, label,_,z_init, s_init, z, s, x_img, mu_0, mu_1 = evaluate(0, model, test_loader, args)
# print(z_init.shape[0])
for i in range(z_init.shape[0]):
    if i in instance_num:
        raw_x = x_img[i,:,:,:].transpose((1,2,0)) * 0.5 + 0.5
        img_list = [raw_x]
        for ii in range(6):
            s[i, 4] = min_val + delta * (ii-1)
            img_list.append(model.decode_x(torch.cat([torch.from_numpy(z[i,:]).unsqueeze(0).cuda().float(), 
                                          torch.from_numpy(s[i,:]).unsqueeze(0).cuda().float()], dim=1))[:, :, 2:30, 2:30].contiguous().squeeze(0).detach().permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
        plot_img = np.concatenate(img_list, axis = 1)
        plt.imshow(plot_img)
        if not os.path.exists('./saved_img_fix_z/'):
            os.makedirs('./saved_img_fix_z/')
        plt.imsave('./saved_img_fix_z/fix_z_%05d.png'%i, plot_img)