import torch
import os
from PIL import Image
import os
import os.path
import numpy as np
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
import scipy.stats as stats
import copy

def generate_mnist_color(sigma = 0.02, env_num=2, color_num = 2, env_type=0, test_ratio = 0.1):
    train_save_dir = './data/colored_MNIST_%0.2f_env_%d_%d_c_%d_%0.2f/train/'%(sigma, env_num, env_type, color_num, test_ratio)
    test_save_dir = './data/colored_MNIST_%0.2f_env_%d_%d_c_%d_%0.2f/test/' % (sigma, env_num, env_type, color_num, test_ratio)
    for i in range(10):
        if not os.path.exists('%s%d/'%(train_save_dir, i)):
            os.makedirs('%s%d/'%(train_save_dir, i))
        if not os.path.exists('%s%d/'%(test_save_dir, i)):
            os.makedirs('%s%d/'%(test_save_dir, i))

    a = datasets.MNIST('./data/', train=True, download=True,
                       transform=None)
    a = datasets.MNIST('./data/', train=False, download=True,
                       transform=None)

    train_dir = './data/MNIST/processed/training.pt'
    test_dir = './data/MNIST/processed/test.pt'
    if color_num == 2:
        colors = {
            0: (0.0, 204.0, 0.0),
            1: (0.0, 204.0, 0.0),
            2: (0.0, 204.0, 0.0),
            3: (0.0, 204.0, 0.0),
            4: (0.0, 204.0, 0.0),
            5: (204.0, 0.0, 0.0),
            6: (204.0, 0.0, 0.0),
            7: (204.0, 0.0, 0.0),
            8: (204.0, 0.0, 0.0),
            9: (204.0, 0.0, 0.0),
        }
    test_color = (0.5, 0.5, 0.5)
    for key in colors.keys():
        colors[key] = colors[key] / np.array([255.0, 255.0, 255.0])
    
    if env_num == 2:
        if env_type == 0:
            train_ratio = [0.9, 0.8]
        elif env_type == 1:
            train_ratio = [0.9, 0.95]
        elif env_type == 3:
            train_ratio = [0.99, 0.95]
        elif env_type == 2:
            train_ratio = [0.9, 0.85]
    
    # training process
    data, target = torch.load(train_dir)
    each_env_num = data.size(0) // env_num
    for i in range(data.size(0)):
        tar = int(target[i])
        img = data[i].numpy()
        img = np.expand_dims(img, 0)
        img = img.repeat(3, 0)

        env_idx = i // each_env_num
        samples = stats.bernoulli.rvs(p=train_ratio[env_idx], size=100)
        if samples[0] == 1:
            mean_color = colors[tar]
        else:
            if color_num == 2:
                mean_color = colors[9 - tar]
        current_color = mean_color + np.random.normal(0.0, sigma, (3,))
        current_color[current_color < 0.0] = 0.0
        current_color[current_color > 1.0] = 1.0
        for s in range(3):
            img[s,:,:] = current_color[s] * img[s,:,:]

        img = Image.fromarray(img.astype('uint8').transpose((1,2,0)))
        img.save('%s%d/img_%05d_%0.2f_%d.png'%(train_save_dir, tar, i, train_ratio[env_idx], env_idx))

    # testing process
    data, target = torch.load(test_dir)

    for i in range(data.size(0)):
        tar = int(target[i])
        img = data[i].numpy()
        img = np.expand_dims(img, 0)
        img = img.repeat(3, 0)

        samples = stats.bernoulli.rvs(p=test_ratio, size=100)
        if samples[0] == 1:
            mean_color = colors[tar]
        else:
            if color_num == 2:
                mean_color = colors[9 - tar]
        current_color = mean_color + np.random.normal(0.0, sigma, (3,))
        current_color[current_color < 0.0] = 0.0
        current_color[current_color > 1.0] = 1.0
        for s in range(3):
            img[s, :, :] = current_color[s] * img[s, :, :]

        img = Image.fromarray(img.astype('uint8').transpose((1, 2, 0)))
        img.save('%s%d/img_%05d_%0.2f_%d.png' % (test_save_dir, tar, i, test_ratio, 0))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--env_num', type=int, default=2)
    parser.add_argument('--env_type', type=int, default=0)
    parser.add_argument('--color_num', type=int, default=2)

    args = parser.parse_args()
    generate_mnist_color(sigma=args.sigma, env_num=args.env_num, env_type=args.env_type, color_num=args.color_num,
                         test_ratio=args.test_ratio)