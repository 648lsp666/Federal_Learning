# 控制全局模型 

import math
import torch 
import numpy as np
from torchvision import datasets, transforms

from utils import *
from pruning_utils import *
from pruning_utils import regroup as prune_regroup
from conf import conf
import torch.nn.utils.prune as prune

class Global_model(object):
  # 初始化的变量在这里面
  def __init__(self) -> None:
    torch.cuda.set_device(int(conf['global_dev']))
    self.device = torch.device(conf['device'])
    # self.model, self.train_loader, self.val_loader, self.test_loader = setup_model_dataset(conf)
    self.model = torch.load(conf['init_model'])
    self.model.cuda()
    self.val_loader=get_dataset(-1)

    self.decreasing_lr = list(map(int, conf['decreasing_lr'].split(',')))
    self.optimizer = torch.optim.SGD(self.model.parameters(), float(conf['lr']), momentum=conf['momentum'],
                                     weight_decay=float(conf['weight_decay']))
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decreasing_lr, gamma=0.1)

    print(self.model.normalize)

    self.global_epoch = conf['global_epoch']
    self.node_num = conf['num_client']  #可以声明对象时按实际需求设置
    self.sub_epoch = conf['local_epoch']
    self.start_ratio = conf['start_ratio'] #初始剪枝率
    self.ratio = self.start_ratio

    self.init_weight = self.model.state_dict()
    #self.load_pretrained()

  #聚合函数.local_weights应当是包含多个model.state_dict()的列表
  def aggregate(self,local_weights):
    self.model.load_state_dict(self.average_weights(local_weights))

  # 每一轮FL非结构剪枝,输入剪枝率@mk,由于中途需要remove_prune,此剪枝率应递增
  def u_prune(self, ratio):
    #unstructure prune
    #print('before unstructure prune:')
    #print(self.model.state_dict().keys())
    #print('Current prune rate:' + str(ratio))
    pruning_model(self.model, ratio, conv1=True)  #全局非结构化剪枝
    #check_sparsity(self.model, conv1=False)

  # 掩码结构化重组
  # 未剪枝模型--(mask_weight结构化重组)->剪枝模型
  def regroup(self, weight_with_mask):
    current_mask = extract_mask(weight_with_mask)
    for key in current_mask:
      mask = current_mask[key]
      shape = current_mask[key].shape
      current_mask[key] = prune_regroup(mask.view(shape[0], -1)).view(*shape)
      #print(current_mask[key].mean())
    prune_model_custom(self.model, current_mask)
    # prune_random_betweeness(model, current_mask, int(args.num_paths), downsample=downsample, conv1=args.conv1)
    #check_sparsity(self.model, conv1=args.conv1)
    return current_mask

  def refill(self, weight_with_mask):
    current_mask = extract_mask(weight_with_mask)

    # 先用train_loader=self.val_loader 把代码跑通
    prune_model_custom_fillback(self.model, current_mask, criteria='remain', train_loader=self.val_loader,trained_weight=self.model.state_dict(),init_weight=self.init_weight)

  # 载入预训练模型(彩票)
  def load_pretrained(self):
    initalization = torch.load(conf['pretrained'], map_location=torch.device(conf['global_dev']))
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

    if not (conf['prune_type'] == 'lt' or conf['prune_type'] == 'trained'):
      keys = list(loading_weight.keys())
      for key in keys:
        if key.startswith('fc') or key.startswith('conv1'):
          del loading_weight[key]

      loading_weight['fc.weight'] = new_initialization['fc.weight']
      loading_weight['fc.bias'] = new_initialization['fc.bias']
      loading_weight['conv1.weight'] = new_initialization['conv1.weight']

    #print('*number of loading weight={}'.format(len(loading_weight.keys())))
    #print('*number of model weight={}'.format(len(self.model.state_dict().keys())))
    self.model.load_state_dict(loading_weight)

  #获取参数平均值
  def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
      for i in range(1, len(w)):
        w_avg[key] += w[i][key]
      w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

  import math

  #降速逼近end_ratio
  def increase_ratio(self, end_ratio, speed=0.2):
    delta = end_ratio - self.ratio
    change = delta * (1 - math.exp(-speed))

    self.ratio += change

  def init_ratio(self):
    self.ratio = self.start_ratio


if __name__=='__main__':
  global_model=Global_model()
  weight_with_mask = global_model.model.state_dict()
  global_model.init_ratio()

  target_ratio = 0.8

  for struct_prune_time in range(2):
    for unstruct_prune_time in range(3):
      print(f'Unstruct Prune[{struct_prune_time}][{unstruct_prune_time}], Prune Ratio: {global_model.ratio}')
      # 非结构化剪枝（可迭代）
      global_model.u_prune(global_model.ratio)
      weight_with_mask = global_model.model.state_dict()
      remove_prune(global_model.model, conv1=True)
      # 如果达到目标剪枝率，跳出循环
      if global_model.ratio == target_ratio:
        print('Reach Target Ratio')
        break
      global_model.increase_ratio(target_ratio)

    print(f'Struct Prune[{struct_prune_time}]')
    #mask_weight = global_model.regroup(weight_with_mask)
    global_model.refill(weight_with_mask)
    remove_prune(global_model.model, conv1=False)
    check_sparsity(global_model.model, conv1=False)
  # 这里写简单的测试@mk
  # 随便剪枝几次 然后regroup一下 ，函数能跑通就行 我后面再调试@zhy