import torch
from models.cnn_cifar import CNNCifar
from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from conf import conf
import time
import torchvision
from torchvision.models import resnet18
import torch_pruning as tp
from pruning_utils import MySlimmingImportance,MySlimmingPruner
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
model=CNNCifar()


inputs = torch.randn(32, 3, 32, 32)  # batch_size张 32x32 的图片
labels = torch.randint(0, 9, (32,))  # 16个标签，范围在0-9之间 10分类任务
loader=torch.utils.data.DataLoader(list(zip(inputs, labels)),batch_size=32)
device='cuda'
# 训练
model.train()
# 创建损失函数和优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
loss_fn = nn.CrossEntropyLoss().to(conf['device'])
stime=time.time()
for e in range(1000):
  running_loss = 0.0
  total=0
  correct=0
  for X,y in loader:
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(time.time()-stime)