import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
])

# 加载训练数据
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True)

# 加载测试数据
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)  # 3个输入通道，6个输出通道，5x5卷积核
#         self.pool = nn.MaxPool2d(2, 2)   # 2x2最大池化
#         self.conv2 = nn.Conv2d(6, 16, 5) # 第二层卷积
#         self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层，120个神经元
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)  # 10个输出类别

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积和池化
#         x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积和池化
#         x = x.view(-1, 16 * 5 * 5)  # 展平层
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class LeNetBN5Cifar(nn.Module):
    def __init__(self, nclasses = 10, cfg=None, ks=5):
        super(LeNetBN5Cifar, self).__init__()
        if cfg == None: 
            self.cfg = [6, 'M', 16, 'M'] 
        else: 
            self.cfg = cfg
    
        self.ks = ks 
        fc_cfg = [120, 84, 100]
        
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)        
        
        self.fc1 = nn.Linear(self.cfg[-2] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nclasses)
        
        self._initialize_weights()
    
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        idx_maxpool = 1 
        idx_bn = 1
        idx_conv = 1 
        idx_relu = 1
        for v in self.cfg:
            if v == 'M':
                layers += [('maxpool{}'.format(idx_maxpool), nn.MaxPool2d(kernel_size=2, stride=2))]
                idx_maxpool += 1 
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [('conv{}'.format(idx_conv), conv2d), ('bn{}'.format(idx_bn), nn.BatchNorm2d(v)),
                               ('relu{}'.format(idx_relu), nn.ReLU(inplace=True))]
                    idx_bn += 1 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1 
                in_channels = v
        
        [self.main.add_module(n, l) for n, l in layers]
    
    def forward(self, x):
        #x = self.main.conv1(x)
        x = self.main(x)
        
        #print(x.shape)
        #print(self.cfg[2])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return

def updateBN(mymodel, args):
    for m in mymodel.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
            
    return 



device = torch.device("cuda:0") 
net = LeNetBN5Cifar().to(device)
net=torch.load('models/init_resnet18.pt')
net=net.to(device)

criterion = nn.CrossEntropyLoss().to('cuda')
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)


num_epochs = 200  # 总的训练轮数
acc=[]
for epoch in range(num_epochs):
    # 训练模型
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 每个epoch结束时评估模型的精确度
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = correct / total
        print(epoch_accuracy)
        acc.append(epoch_accuracy)
print(acc)
print('Finished Training')

import matplotlib.pyplot as plt

# 假设你有一个包含每个epoch精度的列表

epochs = list(range(1, len(acc) + 1))

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 在这里保存图表
# 指定文件路径和名称，例如保存为PNG格式
plt.savefig('resnet18_cifar10_ep200_standalone.png')

# 显示图形
plt.show()