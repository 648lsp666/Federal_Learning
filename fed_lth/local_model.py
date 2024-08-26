

# 该文件包含客户端模型的相关行为 @zhy
# 该类与通信模块分离，仅需要客户端id用于划分本地数据集

import torch
from dataset import get_dataset,get_data_indices
from torch.utils.data import Dataset, DataLoader, TensorDataset,Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from conf import conf
import time
import torchvision
import torch_pruning as tp
from pruning_utils import MySlimmingImportance,MySlimmingPruner
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os

# 管理本地训练行为
class Local_model(object):
  def __init__(self, model):
    self.device=conf['local_dev']
    self.model=copy.deepcopy(model)
    self.noise_scale = self.calculate_noise_scale()
    self.criterion = nn.CrossEntropyLoss().to(self.device)

  def train(self,train_data,train_indices,epoch=conf['local_epoch'],lr=conf['lr'],min_lr=conf['min_lr']):
    lr = max(lr, min_lr)
    print('Train Learning Rate:',lr)
    device=self.device 
    # 训练
    self.model.to(device)
    self.model.train()
    # 创建损失函数和优化器
    optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, min_lr=min_lr, verbose=True)
    train_set=Subset(train_data,train_indices)
    train_loader=DataLoader(train_set,batch_size=conf['batch_size'])
    # 计算时间
    start_time = time.time()
    for e in range(epoch):
      epoch_loss = 0.0
      for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        epoch_loss += loss.item()
        # 梯度裁剪
        if conf['dp_mechanism'] != 'NO':
          self.clip_gradients(self.model)
        optimizer.step()
        # 添加隐私噪声
        if conf['dp_mechanism'] != 'NO':
          self.add_noise(self.model)
      scheduler.step(epoch_loss)
    end_time = time.time()
    elapsed_time = end_time - start_time
    self.model.to('cpu')
    model_para=copy.deepcopy(self.model.state_dict())
    return elapsed_time, model_para
  
  # 梯度裁剪
  def clip_gradients(self, model):
    if conf['dp_mechanism'] == 'Gaussian':
      # Gaussian use 2 norm
      self.per_sample_clip(model, conf['clip'], norm=2)

  def per_sample_clip(self, net, clipping, norm):
    grad_samples = [x.grad_sample for x in net.parameters()]
    per_param_norms = [
      g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
    ]
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
    per_sample_clip_factor = (
      torch.div(clipping, (per_sample_norms + 1e-6))
    ).clamp(max=1.0)
    for grad in grad_samples:
      factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
      grad.detach().mul_(factor.to(grad.device))
    # average per sample gradient after clipping and set back gradient
    for param in net.parameters():
      param.grad = param.grad_sample.detach().mean(dim=0)

  def add_noise(self, model):
    #sensitivity = self.cal_sensitivity(conf['lr'], conf['clip'], self.train_len)
    sensitivity = self.cal_sensitivity(0.01, self.train_len)#0.01)
    state_dict = model.state_dict()
    if conf['dp_mechanism'] == 'Gaussian':
      for k, v in state_dict.items():
        state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.device)
    model.load_state_dict(state_dict)

  
  def cal_sensitivity(lr, clip,  dataset_size):
    #return 2 * lr * clip / dataset_size

    return 2 * 0.01 * clip / dataset_size

  def Gaussian_Simple(delta, epsilon):
    #return np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
  
  
  def calculate_noise_scale(self):
    epsilonsinglequery = 4#conf['epsilon'] / 1#conf['times']
    deltasinglequery = 1e-5#conf['delta'] / 1# conf['times']
    return self.Gaussian_Simple( 4)#(1e-5))#deltasinglequery, epsilonsinglequery )#, epsilonsinglequery)

  def examples(self):
    if conf['dataset_name']=='cifar10':
    # 随机训练测试时间
      inputs = torch.randn(conf['batch_size'], 3, 32, 32)  # batch_size张 32x32 的图片
      labels = torch.randint(0, conf['n_class']-1, (conf['batch_size'],))  # 标签
    elif conf['dataset_name']=='mnist':
      inputs = torch.randn(conf['batch_size'], 1, 28, 28)  # batch_size张 28x28 的图片
      labels = torch.randint(0, conf['n_class']-1, (conf['batch_size'],))  # 标签
    return inputs,labels

  # 随机训练测试时间
  def time_test(self):
    device=self.device
    inputs,labels=self.examples()
    # 训练5 batch，计算平均时间
    self.model.to(self.device)
    self.model.train()
    optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(device)
    # 计算时间
    start_time = time.time()
    for e in range(5):
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

  def s_prune(self,ratio):
    # 剪枝函数 输入是剪枝率
    # self.model.to('cpu')
    imp = MySlimmingImportance()

    # 忽略最后的分类层
    ignored_layers = []
    for m in self.model.modules():
      if isinstance(m, torch.nn.Linear) and m.out_features == 10:
        ignored_layers.append(m)

    # 初始化剪枝器
    inputs,labels=self.examples()
    inputs=inputs.to(self.device)
    iterative_steps = 1
    pruner = MySlimmingPruner(
      self.model,
      inputs,
      importance=imp,
      iterative_steps=iterative_steps,
      ch_sparsity=ratio,
      ignored_layers=ignored_layers,
    )
    pruner.step()

  # 测试剪枝率，输入是时间阈值 单位秒
  def prune_ratio(self,time_T):
    model = self.model
    inputs,labels=self.examples()
    inputs=inputs.to(self.device)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, inputs)
    start = 0.0
    end = 1.0
    step = 0.03
    current = start
    while current <= end:
      self.s_prune(current)
      if self.time_test() <= time_T:
        macs, nparams = tp.utils.count_ops_and_params(model, inputs)
        break
      current += step
    prune_ratio= 1 - nparams/base_nparams
    print(f'prune_ratio:{prune_ratio}')
    # 返回参数稀疏率和通道稀疏率
    return prune_ratio,current
  
  # 评估函数
  def eval(self,test_data,test_indices):
    device=self.device
    self.model.to(device)
    # 加载测试数据集
    test_dataset = Subset(test_data, test_indices)
    # 创建DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    # 将模型设置为评估模式
    self.model.eval()
    # 初始化损失和准确率计数器
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
      for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU
        # 前向传播
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        # 累计损失
        total_loss += loss.item()
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    self.model.to('cpu')
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss,accuracy

#测试代码写在这里面
if __name__=='__main__':
  local=Local_model(torchvision.models.resnet18())
  train,test=get_dataset(False)
  train_idx,test_idx,_=get_data_indices(train,test)
  train_time=local.time_test()
  print(train_time)
  time_T=2.5
  prune_ratio=local.prune_ratio(time_T)
  inputs,labels=local.examples()
  local.train(train,train_idx[0])
  local.eval(test,test_idx[0])
