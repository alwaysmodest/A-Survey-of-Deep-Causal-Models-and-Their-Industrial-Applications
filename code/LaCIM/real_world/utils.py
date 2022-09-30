import os
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import copy
import argparse
import time
import logging
import xlrd
import xlwt
from xlutils.copy import copy as x_copy
import imageio
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from PIL import Image

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
def load_img_path(filepath):
    img = imageio.imread(filepath)
    img_shape = img.shape[1]
    img_3d = img.reshape((img_shape, img_shape, img_shape))
    img_3d = img_3d.transpose((1,2,0))
    return img_3d

def mean_nll(logits, y):
    return F.cross_entropy(logits, y)

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
        if 'mnist' in args.dataset:
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
            if '225. cat_mm8_2-min.png' in self.image_path_list[index]:
                img_1 = np.asarray(img_1.convert('RGBA'))[:, :, :3]
                img_1 = Image.fromarray(img_1.astype('uint8'))
            else:
                img_1 = Image.fromarray(np.asarray(img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
               torch.from_numpy(np.array(self.y[index]).astype('int')), \
               torch.from_numpy(np.array(self.env[index]).astype('int')), \
               torch.from_numpy(np.array(self.u[index]).astype('float32').reshape((1)))
    
    def __len__(self):
        return len(self.image_path_list)


def get_opt():
    parser = argparse.ArgumentParser(description='PyTorch')
    # Model parameters
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',  # test_batch_size������
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='VAE')
    parser.add_argument('--root', type=str, default='/home/botong/Dataset/')
    parser.add_argument('--no-cuda', action='store_true', default=False,  # GPU������Ĭ��ΪFalse
                        help='disables CUDA training')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reg', type=float, default=0.0005)
    parser.add_argument('--reg2', type=float, default=0.00005)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_controler', type=int, default=80)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='AD')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--sample_num', type=int, default=1)

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--zs_dim', type=int, default=256)
    parser.add_argument('--env_num', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--test_ep', type=int, default=50)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--z_ratio', type=float, default=0.5)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args


def make_dirs(args):
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    args.model_save_dir = './results/%s/save_model_%s_%s_%d_%0.1f_%0.1f_lr_%0.4f_%d_%0.4f_wd_%0.4f_%s/'\
                          %(args.dataset, args.model, args.dataset, args.beta, args.alpha,
                            args.lr, args.lr2, args.lr_controler, args.lr_decay, args.reg, current_time)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    return args

def get_logger(opt):
    logger = logging.getLogger('AL')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training.log'.format(opt.model_save_dir))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def compute_acc(pred, target):
    return (np.sum(np.argmax(pred, axis=1) == target).astype('int')) / pred.shape[0]

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

def checkpoint(epoch, save_folder, save_model, is_best=0, other_info = None, logger = None):
    if is_best:
        model_out_tar_path = os.path.join(save_folder, "best_acc.pth.tar")
    else:
        model_out_tar_path = os.path.join(save_folder, "checkpoints.pth.tar")

    torch.save({
        'state_dict':save_model.state_dict(),
        'epoch':epoch,
        'other_info': other_info
    },model_out_tar_path)
    if logger is not None:
        logger.info("Checkpoint saved to {}".format(model_out_tar_path))
    else:
        print("Checkpoint saved to {}".format(model_out_tar_path))

def adjust_learning_rate(optimizer, epoch, lr, lr_decay, lr_controler):
    for param_group in optimizer.param_groups:
        new_lr = lr * lr_decay ** (epoch // lr_controler)
        param_group['lr'] = lr * lr_decay ** (epoch // lr_controler)
    print('current lr is ', new_lr)
        