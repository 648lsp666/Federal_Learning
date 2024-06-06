#生成初始模型文件

import torch
from torchvision import models

model = models.resnet18()
model.fc=torch.nn.Linear(in_features=512,out_features=10,bias=True)
torch.save(model,'init_resnet18.pt')
