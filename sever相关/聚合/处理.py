import torch
import torchvision.models as models

# 创建两个ResNet-18模型实例并加载state_dict
model1 = models.resnet18()
model2 = models.resnet18()
state_dict1 = torch.load('resnet18-5c106cde.pth')
state_dict2 = torch.load('resnet18-5c106cde.pth')
model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)

# 对两个模型的参数求平均
avg_state_dict = {}
for key in model1.state_dict():
    avg_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2

# 创建新模型并加载平均参数
avg_model = models.resnet18()
avg_model.load_state_dict(avg_state_dict)

# 打印新模型的state_dict
print(avg_model.state_dict())
torch.save(avg_model,'save.pt')
torch.save(avg_model.state_dict(),'save_state.pt')
