

# 该文件包含客户端模型的相关行为 @zhy
# 该类与通信模块分离，仅需要客户端id用于划分本地数据集

import torch 
from dataset import get_dataset
from conf import conf
import time
import torchvision
from torchvision.models import resnet18
import torch_pruning as tp
from pruning_utils import MySlimmingImportance,MySlimmingPruner
import torch.nn as nn
import torch.optim as optim
import os

class Local_model(object):
  def __init__(self, id,model):
    self.id=id
    # 从服务器接收到的模型
    # self.model=torch.load(os.path.join(conf['temp_path'],f'client{id}_init_model')) 
    self.model=model
    # 本地数据集的loader
    self.train_data,self.test_data, self.train_dis,_=get_dataset(id)
    self.train_len=sum(self.train_dis)
    self.train_dis=[i/self.train_len for i in self.train_dis]
    print(self.train_dis)
  
  def local_train(self,train_loader,epoch,lr=0.001):
    device=conf['device']
    self.model.train()
    # 训练
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    # 计算时间
    start_time = time.time()
    for e in range(epoch):
      running_loss = 0.0
      for inputs, labels in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        self.model.to(device)
        inputs=inputs.to(device)
        labels=labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
      print(f'epoch:{e+1} loss: {running_loss / 100:.3f}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'train time:{end_time-start_time}')
    return elapsed_time
    


  # 随机训练测试时间
  def time_test(self):
    # 随机训练测试时间
    inputs = torch.randn(conf['batch_size'], 3, 224, 224)  # batch_size张 224x224 的图片
    labels = torch.randint(0, conf['n_class']-1, (conf['batch_size'],))  # 16个标签，范围在0-9之间 10分类任务
    loader=torch.utils.data.DataLoader(list(zip(inputs, labels)),batch_size=conf['batch_size'])
    # 一个batch的时间
    time=self.local_train(loader,5)/5
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
    example_inputs = torch.randn(1, 3, 224, 224)
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
    # inputs = torch.randn(16, 3, 224, 224)  # 16张 224x224 的图片
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
    example_inputs = torch.randn(conf['batch_size'], 3, 224, 224).to(conf['device'])    
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
        self.model = torchvision.models.resnet18()
        break
      current += step
    prune_ratio= 1 - nparams/base_nparams
    print(f'prune_ratio:{prune_ratio}')
    return prune_ratio
    


#测试代码写在这里面
if __name__=='__main__':
  local=Local_model(1,torchvision.models.resnet18())
  train_time=local.time_test()
  print(train_time)
  time_T=2.5
  prune_ratio=local.prune_ratio(time_T)