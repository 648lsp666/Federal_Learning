

# 该文件包含客户端模型的相关行为 @zhy
# 该类与通信模块分离，仅需要客户端id用于划分本地数据集

import torch 
from dataset import get_dataset
from conf import conf
import time
import torchvision
from torchvision.models import resnet18
import torch_pruning as tp
import torch.nn as nn
import torch.optim as optim


class MySlimmingImportance(tp.importance.Importance):
  def __call__(self, group, **kwargs):
    group_imp = []
    for dep, idxs in group:
      layer = dep.target.module
      prune_fn = dep.handler
      if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
        importance_scores = torch.abs(layer.weight.data)  # 使用 L1 范数计算重要性
        group_imp.append(importance_scores)
    if len(group_imp) == 0: return None
    group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
    return group_imp


class MySlimmingPruner(tp.pruner.MetaPruner):
  def regularize(self, model, reg):
    for m in model.modules():
      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
        m.weight.grad.data.add_(reg * torch.sign(m.weight.data))

class Local_model(object):
  def __init__(self, id):
    self.id=id
    self.model=None  #默认为none 等待服务器下发模型
    # 本地数据集
    self.train_data,self.test_data=get_dataset(id)

  # 随机训练测试时间
  def time_test(self):
    # 随机训练测试时间
    inputs = torch.randn(16, 3, 224, 224)  # 16张 224x224 的图片
    labels = torch.randint(0, conf['n_class']-1, (16,))  # 16个标签，范围在0-9之间 10分类任务
    # 测时间 数据集数量/16*time
    self.model.train()
    # 训练
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    # 计算时间
    start_time = time.time()
    # 前向传播
    outputs = self.model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = time.time()

    # 计算并打印时间
    elapsed_time = end_time - start_time
    final_time = len(self.test_data)*elapsed_time/16

    return final_time

  def s_prune(self,ratio):
    # 剪枝函数 输入是剪枝率
    imp = MySlimmingImportance()

    # 忽略最后的分类层
    ignored_layers = []
    for m in self.model.modules():
      if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
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

    # 定义损失函数和优化器（可更改）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    # 模拟训练数据
    inputs = torch.randn(16, 3, 224, 224)  # 16张 224x224 的图片
    labels = torch.randint(0, 1000, (16,))  # 16个标签，范围在0-999之间

    # 进行稀疏训练
    for epoch in range(5):  # 训练5个epoch
      self.model.train()
      optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      pruner.regularize(self.model, reg=1e-5)  # 稀疏化
      optimizer.step()

    pruner.step()

  # 测试剪枝率，输入是时间阈值 单位秒
  def prune_ratio(self,time_T):
    model = self.model
    start = 0.0
    end = 1.0
    step = 0.1
    current = start
    while current <= end:
      self.model = model
      self.s_prune(current)
      if self.time_test() <= time_T:
        self.model = model
        macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        prune_ratio = (base_nparams - nparams) / base_nparams * 100
        break
      current += step
    return prune_ratio
    


#测试代码写在这里面
if __name__=='__main__':
  local=Local_model(1)
  local.model=torchvision.models.resnet18()
  train_time=local.time_test()
  print(train_time)
  time_T=1000
  prune_ratio=local.prune_ratio(time_T)
  print(prune_ratio)

  #剪枝测试
  example_inputs = torch.randn(1, 3, 224, 224)
  base_macs, base_nparams = tp.utils.count_ops_and_params(local.model, example_inputs)
  # 执行剪枝
  local.s_prune(0.5)
  # 剪枝后计算模型参数和计算量
  macs, nparams = tp.utils.count_ops_and_params(local.model, example_inputs)
  print("参数数量: {:.2f} M => {:.2f} M".format(base_nparams / 1e6, nparams / 1e6))
  print("计算量: {:.2f} G => {:.2f} G".format(base_macs / 1e9, macs / 1e9))