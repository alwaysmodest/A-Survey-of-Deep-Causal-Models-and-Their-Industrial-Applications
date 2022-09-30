# coding:utf8
from __future__ import print_function
import torch.optim as optim
from torch import nn, optim, autograd
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
from models import *
import torch.nn.functional as F
import torch.nn as nn
import sys

def mean_nll(logits, y):
    return F.nll_loss(torch.log(logits), y)

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def VAE_loss_prior(recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args):
    """
    pred_y: predicted y
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    q_y_s: prior
    beta: tradeoff params
    """
    eps = 1e-5
    if 'mnist' in args.dataset:
        x = x * 0.5 + 0.5
        BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), x.view(-1, 3 * 28 ** 2), reduction='mean')
    
    KLD_element = torch.log(logvar_prior.exp() ** 0.5 / logvar.exp() ** 0.5) + \
                  0.5 * ((mu - mu_prior).pow(2) + logvar.exp()) / logvar_prior.exp() - 0.5
    KLD = KLD_element.mul_(1).mean()
    
    return BCE, KLD


def train(epoch, model, optimizer, dataloader, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    args.fix_mu = 1
    args.fix_var = 1
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        loss = torch.FloatTensor([0.0]).cuda()

        recon_loss = torch.FloatTensor([0.0]).cuda()
        kld_loss = torch.FloatTensor([0.0]).cuda()
        cls_loss = torch.FloatTensor([0.0]).cuda()
        # irm_loss = torch.FloatTensor([0.0]).cuda()
        for ss in range(args.env_num):
            if torch.sum(env == ss) <= 1:
                continue
            _, recon_x, mu, logvar, mu_prior, logvar_prior, z, s, zs = model(x[env == ss,:,:,:], ss, feature=1, is_train = 1)
            pred_y = model.get_y_by_zs(mu, logvar, ss)
            #pred_y = model.get_pred_y(x[env == ss,:,:,:], ss)
            #print(recon_x.size(), x[env == ss,:,:,:].size())
            recon_loss_t, kld_loss_t = VAE_loss_prior(recon_x, x[env == ss,:,:,:], mu, logvar, mu_prior, logvar_prior, zs, args)
            cls_loss_t = F.nll_loss(torch.log(pred_y), target[env == ss])
            # irm_loss_t = penalty(pred_y, target[env == ss])
            
            accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target[env == ss].detach().cpu().numpy()),
                            pred_y.size(0))
            recon_loss = torch.add(recon_loss, torch.sum(env == ss) * recon_loss_t)
            kld_loss = torch.add(kld_loss, torch.sum(env == ss) * kld_loss_t)
            cls_loss = torch.add(cls_loss, torch.sum(env == ss) * cls_loss_t)
            # irm_loss = torch.add(irm_loss, torch.sum(env == ss) * irm_loss_t)
        recon_loss = recon_loss / x.size(0)
        kld_loss = kld_loss / x.size(0)
        cls_loss = cls_loss / x.size(0)
        # irm_loss = irm_loss / x.size(0)

        RECON_loss.update(recon_loss.item(), x.size(0))
        KLD_loss.update(kld_loss.item(), x.size(0))
        classify_loss.update(cls_loss.item(), x.size(0))
        loss = torch.add(loss, args.alpha * recon_loss + args.beta * kld_loss + args.gamma * cls_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss.update(loss.item(), x.size(0))
        
        
        if batch_idx % 10 == 0:
            args.logger.info(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg,
                        KLD_loss.avg * args.beta,
                        classify_loss.avg,
                        all_loss.avg,
                        accuracy.avg * 100))
    
    if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' or \
            args.model == 'VAE_f' or args.model == 'sVAE_f':
        all_zs = all_zs[:batch_begin]
    args.logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, all_loss.avg))
    
    return all_zs, accuracy.avg

def evaluate(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        pred_y_init, pred_y = model(x, is_train = 0, is_debug=1)
        pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()). \
                                              reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
        accuracy_init.update(compute_acc(np.array(pred_y_init.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))

        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    return pred, accuracy.avg


def main():
    args = get_opt()
    
    args = make_dirs(args)
    logger = get_logger(args)
    logger.info(str(args))
    args.logger = logger
    other_info = {}
    
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D(root = args.root, args=args, fold='train',
                                                 transform=transforms.Compose([
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last = True)
        test_loader = DataLoader(get_dataset_2D(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)
        val_loader = None

    model = d_LaCIM(in_channel=args.in_channel,
                     zs_dim=args.zs_dim,
                     num_classes=args.num_classes,
                     decoder_type=1,
                     total_env=args.env_num,
                     args=args,
                     ).cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params, '%0.4f M' % (pytorch_total_params / 1e6))


    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, args)

        is_best = 0
        pred_test, test_acc = evaluate(epoch, model, test_loader, args)
        if test_acc >= best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
            logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
                        % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))

        checkpoint(epoch, args.model_save_dir, model, is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d'
                    %(epoch, test_acc, best_acc, best_acc_ep))
    logger.info('model save path: %s'%args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()