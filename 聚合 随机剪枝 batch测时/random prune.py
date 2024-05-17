import torch
import torch.nn.utils.prune as prune
import torchvision.models as models

# 创建ResNet-18模型实例并加载state_dict
model1 = models.resnet18()
state_dict1 = torch.load('resnet18-5c106cde.pth')
model1.load_state_dict(state_dict1)

# 定义一个函数，用于对模型的所有卷积层进行L1非结构化剪枝
def apply_l1_unstructured_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # 对bias进行同样的剪枝
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)

# 对模型进行随机剪枝
apply_l1_unstructured_pruning(model1, amount=0.2)

# 可以查看剪枝后的模型结构
print(model1)

# 如果需要移除剪枝掩码并压缩模型，可以使用以下代码
def remove_pruning_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
            if module.bias is not None:
                prune.remove(module, 'bias')

# 移除剪枝掩码
remove_pruning_masks(model1)

# 查看剪枝后的模型参数
print(model1)

