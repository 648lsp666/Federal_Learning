# 控制全局模型 

import torch 
import numpy as np
from torchvision import datasets, transforms

from conf import conf


class Global_model(object):
  # 初始化的变量在这里面
  def __init__(self) -> None:
<<<<<<< Updated upstream
    self.init_model=torch.load(conf['init_model'])
  
  #聚合函数 
  def aggregate(self):
    pass
=======
    self.device = torch.device(conf['global_dev'])
    self.model, self.train_loader, self.val_loader, self.test_loader = setup_model_dataset(conf)
    self.model.to(self.device)

    #对于不参与实际训练的服务端，以下是否无需定义？
    self.decreasing_lr = list(map(int, conf['decreasing_lr'].split(',')))
    self.optimizer = torch.optim.SGD(self.model.parameters(), conf['lr'], momentum=conf['momentum'],
                                     weight_decay=conf['weight_decay'])
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decreasing_lr, gamma=0.1)

    print(self.model.normalize)
    self.global_epoch = conf['global_epoch']
    self.node_num = conf['num_client']  #可以声明对象时按实际需求设置
    self.sub_epoch = conf['local_epoch']

    self.load_pretrained()
>>>>>>> Stashed changes

  #聚合函数.local_weights应当是包含多个model.state_dict()的列表
  def aggregate(self,local_weights):
    self.model.load_state_dict(self.average_weights(local_weights))

  # 每一轮FL非结构剪枝,输入剪枝率@mk,由于中途需要remove_prune,此剪枝率应递增
  def u_prune(self,ratio):
    #unstructure prune
    #print('before unstructure prune:')
    #print(self.model.state_dict().keys())
    #print('Current prune rate:' + str(ratio))
    pruning_model(self.model, ratio, conv1=True)  #全局非结构化剪枝
    #check_sparsity(self.model, conv1=False)


  # 掩码结构化重组@mk,注意：调用前应当是orig+mask参数字典，调用后返回依旧为orig+mask参数字典，请手动remove_prune
  def regroup(self):
    mask_weight = self.model.state_dict()  # Keep Current unstructure pruning
    #loading_weight = extract_main_weight(self.model.state_dict(), fc=True, conv1=True)
    remove_prune(self.model, conv1=True)  # remove mask and prune permanently
    #print('after remove prune:')
    #print(self.model.state_dict().keys())

    current_mask = extract_mask(mask_weight)
    for key in current_mask:
      mask = current_mask[key]
      shape = current_mask[key].shape
      current_mask[key] = regroup(mask.view(shape[0], -1)).view(*shape)
      #print(current_mask[key].mean())
    prune_model_custom(self.model, current_mask)
    # prune_random_betweeness(model, current_mask, int(args.num_paths), downsample=downsample, conv1=args.conv1)
    #check_sparsity(self.model, conv1=args.conv1)

<<<<<<< Updated upstream
=======
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

>>>>>>> Stashed changes
if __name__=='__main__':
  global_model=Global_model()
  test_rate = 0.2
  for i in range(3):
    global_model.u_prune(test_rate)
    remove_prune(global_model.model, conv1=False) #建议还是remove一下，方便功能拓展
    check_sparsity(global_model.model, conv1=False)
    test_rate += 0.2
  global_model.regroup()
  remove_prune(global_model.model, conv1=False)
  check_sparsity(global_model.model, conv1=False)
  # 这里写简单的测试@mk
  # 随便剪枝几次 然后regroup一下 ，函数能跑通就行 我后面再调试@zhy