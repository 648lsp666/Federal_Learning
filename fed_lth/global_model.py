# 控制全局模型 
import copy
import math
import torch
import torch_pruning as tp
import numpy as np
from torchvision import datasets, transforms

# from utils import *
from pruning_utils import *
from pruning_utils import regroup as prune_regroup
from conf import conf
import torch.nn.utils.prune as prune
from dataset import get_dataset

  #获取参数平均值
def average_weights(w):
  w_avg = copy.deepcopy(w[0])
  for key in w_avg.keys():
    for i in range(1, len(w)):
      if key[-5:]=='_mask':
        continue
      w_avg[key] += w[i][key]
    w_avg[key] = torch.div(w_avg[key], len(w))
  return w_avg

class Global_model(object):
  # 初始化的变量在这里面
  def __init__(self) -> None:
    # torch.cuda.set_device(int(conf['global_dev']))
    self.device = torch.device(conf['device'])
    # self.model, self.train_loader, self.val_loader, self.test_loader = setup_model_dataset(conf)
    self.model = torch.load(conf['init_model'])
    self.model.to(self.device)
    self.train_loader,self.val_loader=get_dataset(-1)

    self.decreasing_lr = list(map(int, conf['decreasing_lr'].split(',')))
    self.optimizer = torch.optim.SGD(self.model.parameters(), float(conf['lr']), momentum=conf['momentum'],
                                     weight_decay=float(conf['weight_decay']))
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decreasing_lr, gamma=0.1)

    self.node_num = conf['num_client']  #可以声明对象时按实际需求设置
    self.sub_epoch = conf['local_epoch']
    self.start_ratio = conf['start_ratio'] #初始剪枝率
    self.ratio = self.start_ratio

    self.init_weight = self.model.state_dict()
    #self.load_pretrained()
    self.acc=[]
    self.loss=[]
    self.global_epoch=0

  #聚合函数.local_weights应当是包含多个model.state_dict()的列表
  def aggregate(self,local_weights,client_loss,client_acc):
    self.global_epoch+=1
    self.model.load_state_dict(average_weights(local_weights))
    avg_loss=sum(client_loss)/len(client_loss)
    avg_acc=sum(client_acc)/len(client_acc)
    self.acc.append(avg_acc)
    self.loss.append(avg_loss)
    print(f'avg_loss:{avg_loss},acc:{avg_acc}')
    

  # 每一轮FL非结构剪枝,输入剪枝率@mk,由于中途需要remove_prune,此剪枝率应递增
  def u_prune(self, ratio):
    #unstructure prune
    #print('before unstructure prune:')
    #print(self.model.state_dict().keys())
    #print('Current prune rate:' + str(ratio))
    pruning_model(self.model, ratio, conv1=False)  #全局非结构化剪枝
    #check_sparsity(self.model, conv1=False)

  #Torch_pruning结构化剪枝,imp_strategy=Magnitude/Taylor(Default)/Hessian
  def tp_prune(self, trace_input, ratio,
               imp_strategy='Taylor', degree=2, iterative_steps=1, show_step=False, show_group=False):
    print(f'Channel Ratio:{ratio}')
    if imp_strategy == 'Magnitude':
      assert degree == 1 or degree == 2  # degree must be 1 or 2
      imp = tp.importance.MagnitudeImportance(p=degree)
    elif imp_strategy == 'Taylor':
      imp = tp.importance.TaylorImportance()
    elif imp_strategy == 'Hessian':
      imp = tp.importance.HessianImportance()
    else:
      return

    # Ignore some layers, e.g., the output layer
    ignored_layers = []
    for m in self.model.modules():
      if isinstance(m, torch.nn.Linear) and m.out_features == conf['num_class']:
        ignored_layers.append(m)  # DO NOT prune the final classifier!

    # Initialize a pruner
    pruner = tp.pruner.MagnitudePruner(
      self.model,
      trace_input,
      importance=imp,
      iterative_steps=iterative_steps,
      pruning_ratio=ratio,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
      ignored_layers=ignored_layers,
    )

    # prune the model, iteratively if necessary.
    base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, trace_input)
    macs, nparams = base_macs, base_nparams
    for i in range(iterative_steps):
      if isinstance(imp, tp.importance.TaylorImportance):
        # A dummy loss, please replace it with your loss function and data!
        loss = self.model(trace_input).sum()
        loss.backward()  # before pruner.step()
      if show_group:
        for group in pruner.step(
                interactive=True):  # Warning: groups must be handled sequentially. Do not keep them as a list.
          print(group)
          group.prune()
      else:
        pruner.step()
      macs, nparams = tp.utils.count_ops_and_params(self.model, trace_input)
      if show_step:
        print('Current sparsity:' + str(100 * nparams / base_nparams) + '%')

    return nparams/base_nparams

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

  def refill(self, weight_with_mask,mask_only=False):
    current_mask = extract_mask(weight_with_mask)
    # 先用train_loader=self.val_loader 把代码跑通
    if mask_only:
      return prune_model_custom_fillback(self.model, current_mask, criteria='remain', train_loader=self.val_loader,trained_weight=self.model.state_dict(),init_weight=self.init_weight,return_mask_only=True)
    else:
      return prune_model_custom_fillback(self.model, current_mask, criteria='remain', train_loader=self.val_loader,trained_weight=self.model.state_dict(),init_weight=self.init_weight)
  # 载入预训练模型(彩票)
  def load_pretrained(self):
    initalization = torch.load(conf['pretrained'], map_location=torch.device(conf['global_dev']))
    if 'init_weight' in initalization.keys():
      print('loading from init_weight')
      initalization = initalization['init_weight']
    elif 'state_dict' in initalization.keys():
      print('loading from state_dict')
      initalization = initalization['state_dict']

    loading_weight = extract_main_weight(initalization)
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

  #降速逼近end_ratio，最低2%
  def increase_ratio(self, end_ratio, speed=0.2):
    delta = end_ratio - self.ratio
    change = delta * (1 - math.exp(-speed))
    if change <0.05:
      change=0.05
    self.ratio += change

  def init_ratio(self):
    self.ratio = self.start_ratio

  def get_bits(self):
    state_dict = self.model.state_dict()
    total_bits = 0
    for param_name, param_tensor in state_dict.items():
      total_bits += param_tensor.numel() * param_tensor.element_size() * 8
    return total_bits

    # 评估函数
  def eval(self):
    device=conf['device']
    self.model.eval()  # 启用测试模式
 
    # 初始化测试精度
    correct = 0
    total = 0
    
    # 通过测试数据集
    with torch.no_grad():  # 不追踪梯度信息，加速测试
      for images, labels in self.val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算精度
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy}')
    return test_accuracy

if __name__=='__main__':
  global_model=Global_model()
  weight_with_mask = global_model.model.state_dict()
  global_model.init_ratio()
  target_ratio = 0.5
  # Refill/Regroup prune
  # for struct_prune_time in range(2):
  #   for unstruct_prune_time in range(3):
  #     print(f'Unstruct Prune[{struct_prune_time}][{unstruct_prune_time}], Prune Ratio: {global_model.ratio}')
  #     # 非结构化剪枝（可迭代）
  #     global_model.u_prune(global_model.ratio)
  #     weight_with_mask = global_model.model.state_dict()
  #     remove_prune(global_model.model, conv1=True)
  #     # 如果达到目标剪枝率，跳出循环
  #     if global_model.ratio == target_ratio:
  #       print('Reach Target Ratio')
  #       break
  #     global_model.increase_ratio(target_ratio)
  #
  #   print(f'Struct Prune[{struct_prune_time}]')
  #   #mask_weight = global_model.regroup(weight_with_mask)
  #   global_model.refill(weight_with_mask)
  #   remove_prune(global_model.model, conv1=False)
  #   check_sparsity(global_model.model, conv1=False)

  # Refill + Torch_pruning prune
  rewind_weight = global_model.model.state_dict()
  # train code here
  if True:
    for unstruct_prune_time in range(3):
      print(f'Unstruct Prune[{unstruct_prune_time}], Prune Ratio: {global_model.ratio}')
      # 非结构化剪枝（可迭代）
      global_model.u_prune(global_model.ratio)
      weight_with_mask = global_model.model.state_dict()
      # remove_prune(global_model.model, conv1=False)
      # 如果达到目标剪枝率，跳出循环
      if global_model.ratio == target_ratio:
        print('Reach Target Ratio')
        break
      global_model.increase_ratio(target_ratio)
    remove_prune(global_model.model, conv1=False)
    print(f'Refill Struct Prune')
    #mask_weight = global_model.regroup(weight_with_mask)
    prune_mask = global_model.refill(weight_with_mask, mask_only=True)
    #remove_prune(global_model.model, conv1=False)
    check_sparsity(global_model.model, conv1=False)

  # Recover weight
  global_model.model.load_state_dict(rewind_weight)

  # Apply Sparity to model
  prune_model_custom(global_model.model,prune_mask, conv1=False)
  remove_prune(global_model.model, conv1=False)

  # TP Permenant Prune
  global_model.model.zero_grad()  # We don't want to store gradient information
  trace_input , _ = next(iter(global_model.train_loader))
  #bits = global_model.get_bits()
  print('Final sparsity:'+str(100 *
                              global_model.tp_prune(trace_input.to(global_model.device),
                                                    0.3, imp_strategy='Magnitude',
                                                    degree=1)
                              )+'%')
  global_model.train()
  
