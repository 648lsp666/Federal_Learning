

# 该文件包含客户端模型的相关行为 @zhy
# 该类与通信模块分离，仅需要客户端id用于划分本地数据集

import torch
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



class Local_model(object):
  def __init__(self, id,model):
    self.id=id
    # 从服务器接收到的模型
    # self.model=torch.load(os.path.join(conf['temp_path'],f'client{id}_init_model')) 
    self.model=model
    self.device=conf['local_dev']
    # 本地数据集的loader
    self.train_data,self.test_data, self.train_dis,_=get_dataset(id)
    self.train_len=sum(self.train_dis)
    self.train_dis=[i/self.train_len for i in self.train_dis]
    #self.data_size = len(self.train_data)
    self.noise_scale = self.calculate_noise_scale()
    #self.torch_dataset = TensorDataset(torch.tensor(data[0]),
    #                                  torch.tensor(data[1]))
    self.acc=[]
    self.loss=[]
    self.criterion = nn.CrossEntropyLoss()
    # print(self.train_dis)
  
  def local_train(self,train_loader,epoch,lr=conf['lr']):
    device=self.device
    # 训练
    self.model.train()
    # 创建损失函数和优化器
    optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.5)

    # 计算时间
    start_time = time.time()
    for e in range(epoch):
      running_loss = 0.0
      total=0
      correct=0
      for inputs, labels in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        self.model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪
        if conf['dp_mechanism'] != 'NO':
          self.clip_gradients(self.model)
        optimizer.step()
        # 添加隐私噪声
        if conf['dp_mechanism'] != 'NO':
          self.add_noise(self.model)
        # print statistics
        running_loss += loss.item()
              # 计算精度
      test_accuracy = correct / total
      # print(f'epoch:{e+1} loss: {running_loss / 100:.3f}; Test Accuracy: {test_accuracy:.4f}')
      self.loss.append(running_loss/ 100)
      self.acc.append(test_accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'train time:{elapsed_time},acc:{test_accuracy}')
    return elapsed_time,running_loss,test_accuracy
  
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

  # 随机训练测试时间
  def time_test(self):
    # 随机训练测试时间
    inputs = torch.randn(conf['batch_size'], 3, 32, 32)  # batch_size张 32x32 的图片
    labels = torch.randint(0, conf['n_class']-1, (conf['batch_size'],))  # 16个标签，范围在0-9之间 10分类任务
    loader=torch.utils.data.DataLoader(list(zip(inputs, labels)),batch_size=conf['batch_size'])
    # 一个batch的时间
    time, _ ,_ =self.local_train(loader,5)
    time=time/5
    # 计算总时间    
    final_time = self.train_len/conf['batch_size']*time*conf['local_epoch']
    print(f'final_time:{final_time}')
    return final_time

  def s_prune(self,ratio):
    # 剪枝函数 输入是剪枝率
    self.model.to('cpu')
    imp = MySlimmingImportance()

    # 忽略最后的分类层
    ignored_layers = []
    for m in self.model.modules():
      if isinstance(m, torch.nn.Linear) and m.out_features == 10:
        ignored_layers.append(m)

    # 初始化剪枝器
    example_inputs = torch.randn(1, 3, 32, 32)
    iterative_steps = 1
    pruner = MySlimmingPruner(
      self.model,
      example_inputs,
      importance=imp,
      iterative_steps=iterative_steps,
      ch_sparsity=ratio,
      ignored_layers=ignored_layers,
    )

    # # 定义损失函数和优化器（可更改）
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    # # 模拟训练数据
    # inputs = torch.randn(16, 3, 32, 32)  # 16张 32x32 的图片
    # labels = torch.randint(0, 1000, (16,))  # 16个标签，范围在0-999之间

    # # 进行稀疏训练
    # for epoch in range(5):  # 训练5个epoch
    #   self.model.train()
    #   optimizer.zero_grad()
    #   outputs = self.model(inputs)
    #   loss = criterion(outputs, labels)
    #   loss.backward()
    #   pruner.regularize(self.model, reg=1e-5)  # 稀疏化
    #   optimizer.step()

    pruner.step()

  # 测试剪枝率，输入是时间阈值 单位秒
  def prune_ratio(self,time_T):
    model = self.model
    example_inputs = torch.randn(conf['batch_size'], 3, 32, 32).to(self.device)    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    start = 0.0
    end = 1.0
    step = 0.03
    current = start
    while current <= end:
      self.model = model
      self.s_prune(current)
      if self.time_test() <= time_T:
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        break
      current += step
    prune_ratio= 1 - nparams/base_nparams
    print(f'prune_ratio:{prune_ratio}')
    # 返回参数稀疏率和通道稀疏率
    return prune_ratio,current
  
  # 评估函数
  def eval(self):
    device=self.device
    self.model.eval()  # 启用测试模式
 
    # 初始化测试精度
    correct = 0
    total = 0
    
    # 通过测试数据集
    with torch.no_grad():  # 不追踪梯度信息，加速测试
      for images, labels in self.test_data:
        images, labels = images.to(device), labels.to(device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算精度
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy}')
    return test_accuracy

#测试代码写在这里面
if __name__=='__main__':
  local=Local_model(1,torchvision.models.resnet18())
  train_time=local.time_test()
  print(train_time)
  time_T=2.5
  prune_ratio=local.prune_ratio(time_T)