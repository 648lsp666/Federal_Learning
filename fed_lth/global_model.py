# 控制全局模型 

import torch 
import numpy as np
from torchvision import datasets, transforms

from conf import conf


class Global_model(object):
  # 初始化的变量在这里面
  def __init__(self) -> None:
    self.init_model=torch.load(conf['init_model'])
  
  #聚合函数 
  def aggregate(self):
    pass

  # 每一轮FL非结构剪枝,输入剪枝率@mk
  def u_prune(self,ratio):
    pass


  # 掩码结构化重组@mk
  def regroup(self):
    pass

if __name__=='__main__':
  global_model=Global_model()
  for i in range(3):
    global_model.u_prune(0.2)
  global_model.regroup()
  # 这里写简单的测试@mk
  # 随便剪枝几次 然后regroup一下 ，函数能跑通就行 我后面再调试@zhy
  pass