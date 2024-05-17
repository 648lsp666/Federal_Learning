# 控制全局模型 

import torch 
import numpy as np
from torchvision import datasets, transforms

from utils import *
from pruning_utils import *
from pruning_utils import regroup
from conf import conf


class Global_model(object):
  # 初始化的变量在这里面
  def __init__(self) -> None:
    self.device = torch.device(conf['global_dev'])
    self.model, self.train_loader, self.val_loader, self.test_loader = setup_model_dataset(conf)
    self.model.to(self.device)
    self.decreasing_lr = list(map(int, conf['decreasing_lr'].split(',')))
    self.optimizer = torch.optim.SGD(self.model.parameters(), conf['lr'], momentum=conf['momentum'],
                                     weight_decay=conf['weight_decay'])
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decreasing_lr, gamma=0.1)

    print(self.model.normalize)
    self.global_epoch = conf['global_epoch']
    self.node_num = conf['num_client']  #可以声明对象时按实际需求设置
    self.sub_epoch = conf['local_epoch']

    self.load_pretrained()
  #聚合函数
  def aggregate(self):
    pass

  # 每一轮FL非结构剪枝,输入剪枝率@mk
  def u_prune(self,ratio):
    pass


  # 掩码结构化重组@mk
  def regroup(self):
    pass

  # 载入预训练模型(彩票)
  def load_pretrained(self):
    initalization = torch.load(conf['pretrained'], map_location=self.device)
    if 'init_weight' in initalization.keys():
      print('loading from init_weight')
      initalization = initalization['init_weight']
    elif 'state_dict' in initalization.keys():
      print('loading from state_dict')
      initalization = initalization['state_dict']

    loading_weight = extract_main_weight(initalization, fc=True, conv1=True)
    new_initialization = self.model.state_dict()
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


if __name__=='__main__':
  global_model=Global_model()
  for i in range(3):
    global_model.u_prune(0.2)
  global_model.regroup()
  # 这里写简单的测试@mk
  # 随便剪枝几次 然后regroup一下 ，函数能跑通就行 我后面再调试@zhy
  pass