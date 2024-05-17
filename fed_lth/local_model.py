

# 该文件包含客户端模型的相关行为 @zhy

import torch 
from dataset import get_dataset
from conf import conf

class Local_model(object):
  def __init__(self, id):
    self.id=id
    self.model=None  #默认为none 等待服务器下发模型
    # 本地数据集
    self.train_data,self.test_data=get_dataset(id)

  def time_test():
    # 随机训练测试时间
    inputs = torch.randn(16, 3, 224, 224)  # 16张 224x224 的图片
    labels = torch.randint(0, conf['n_class']-1, (16,))  # 16个标签，范围在0-9之间 10分类任务
    



