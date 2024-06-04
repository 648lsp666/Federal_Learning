# @zhy 暂时没用


import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_model(name="vgg16"):
  if name == 'CNNMnist':
    model = CNNMnist()
  if name == "resnet18":
    model = models.resnet18()
    model.fc=nn.Linear(in_features=512,out_features=10,bias=True)
  elif name == "resnet50":
    model = models.resnet50()	
  elif name == "densenet121":
    model = models.densenet121()		
  elif name == "alexnet":
    model = models.alexnet()
  elif name == "vgg16":
    model = models.vgg16()
  elif name == "vgg19":
    model = models.vgg19()
  elif name == "inception_v3":
    model = models.inception_v3()
  elif name == "googlenet":		
    model = models.googlenet()
  
  if torch.cuda.is_available():
    return model.cuda()
  else:
    return model 