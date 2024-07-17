import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True,pin_memory=True)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False,pin_memory=True)
inputs = torch.randn(32, 3, 32, 32)  # batch_size张 32x32 的图片
labels = torch.randint(0,9, (32,))  # 16个标签，范围在0-9之间 10分类任务
loader=torch.utils.data.DataLoader(list(zip(inputs, labels)),batch_size=32)

device='cuda:0'

model = torchvision.models.resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


losslist=[]
acc=[]
# 训练
model.train()
# 创建损失函数和优化器
def train():
    # 计算时间
    start_time = time.time()
    for e in range(1000):
        running_loss = 0.0
        total=0
        correct=0
        for inputs, labels in loader:
        # get the inputs; data is a list of [inputs, labels]
            model.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
        # print statistics
            running_loss += loss.item()
                # 计算精度
        test_accuracy = correct / total
        # print(f'epoch:{e+1} loss: {running_loss / 100:.3f}; Test Accuracy: {test_accuracy:.4f}')
        losslist.append(running_loss/ 100)
        acc.append(test_accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'train time:{elapsed_time},acc:{test_accuracy}')
for i in range(10):
    train()