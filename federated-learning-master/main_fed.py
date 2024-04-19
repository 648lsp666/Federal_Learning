#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import os
import time
import random
import shutil
import argparse
from copy import deepcopy

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd

from utils import *
from pruning_utils import regroup
from pruning_utils_2 import *
from pruning_utils_unprune import *
from pruning_utils import prune_model_custom_fillback

def structure_prune(model, pruning_ratio=0.1):
    # mask
    if args.mask_dir:
        print('loading mask')
        current_mask_weight = torch.load(args.mask_dir, map_location=torch.device('cuda:' + str(args.gpu)))
        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']
        current_mask = extract_mask(current_mask_weight)

        checkpoint = torch.load(args.pretrained, map_location=torch.device('cuda:' + str(args.gpu)))

        for key in current_mask:
            mask = current_mask[key]
            shape = current_mask[key].shape
            current_mask[key] = regroup(mask.view(shape[0], -1)).view(*shape)
            print(current_mask[key].mean())
        prune_model_custom(model, current_mask)
        # prune_random_betweeness(model, current_mask, int(args.num_paths), downsample=downsample, conv1=args.conv1)
        check_sparsity(model, conv1=args.conv1)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.cuda.set_device(int(args.gpu))
    # load dataset and split users

    #using 'cifar10'(args.batch_size,args.data)
    classes = 10
    train_number = 45000
    normalization = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size=args.batch_size, data_dir=args.data)

    # using 'res18'(args.use_sparse_conv)
    model = resnet18(num_classes=classes, imagenet='cifar10' == 'tiny-imagenet', use_sparse_conv=args.use_sparse_conv) #?
    model.cuda()
    criterion = nn.CrossEntropyLoss()

    #load pretrained(args.pretrained,args.gpu,args.use_sparse_conv,args.prune_type)
    initalization = torch.load(args.pretrained, map_location=torch.device('cuda:' + str(args.gpu)))
    ##load init_weight instead of state_dict(if available)
    if 'init_weight' in initalization.keys():
        print('loading from init_weight')
        initalization = initalization['init_weight']
    elif 'state_dict' in initalization.keys():
        print('loading from state_dict')
        initalization = initalization['state_dict']

    loading_weight = extract_main_weight(initalization, fc=True, conv1=True)
    new_initialization = model.state_dict()
    if not 'normalize.std' in loading_weight:
        loading_weight['normalize.std'] = new_initialization['normalize.std']
        loading_weight['normalize.mean'] = new_initialization['normalize.mean']

    if not (args.prune_type == 'lt' or args.prune_type == 'trained'):
        keys = list(loading_weight.keys())
        for key in keys:
            if key.startswith('fc') or key.startswith('conv1'):
                del loading_weight[key]

        loading_weight['fc.weight'] = new_initialization['fc.weight']
        loading_weight['fc.bias'] = new_initialization['fc.bias']
        loading_weight['conv1.weight'] = new_initialization['conv1.weight']

    print('*number of loading weight={}'.format(len(loading_weight.keys())))
    print('*number of model weight={}'.format(len(model.state_dict().keys())))
    model.load_state_dict(loading_weight)

    #define opt(args.decreasing_lr,args.lr,args.momentum,args.weight_decay)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    all_result = {}
    all_result['train'] = []
    all_result['test_ta'] = []
    all_result['ta'] = []

    #start trainning
    start_epoch = 0
    if args.mask_dir:
        remain_weight = check_sparsity(model, conv1=args.conv1)

    model.train()

    # copy weights
    w_glob = model.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

