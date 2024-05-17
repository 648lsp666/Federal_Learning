

# 该文件包含客户端模型的相关行为 @zhy
# 该类与通信模块分离，仅需要客户端id用于划分本地数据集

import torch 
from dataset import get_dataset
from conf import conf
import time
import torchvision

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
      # self.model.train()
    # 训练

    return 'time'

  def s_prune(self,ratio):
    # 剪枝函数 输入是剪枝率
    pass

  # 测试剪枝率，输入是时间阈值 单位秒
  def prune_ratio(self,time_T):
    # 循环
    # 重置模型 self.model
    #prune(ratio) 剪枝 
    #self.time_test() 测时间
    return 'prune_ratio'
    


#测试代码写在这里面
if __name__=='__main__':
  local=Local_model(1)
  local.model=torchvision.models.resnet18()
  train_time=local.time_test()
  time_T=1000
  prune_ratio=local.prune_ratio(time_T)