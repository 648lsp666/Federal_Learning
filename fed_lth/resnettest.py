import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time

inputs = torch.randn(32, 3, 32, 32) 
labels = torch.randint(0,9, (32,)) 
device='cuda:0'
model = torchvision.models.resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()
# 计算时间
start_time = time.time()
for e in range(1000):
    model.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
end_time = time.time()
elapsed_time = end_time - start_time
print(f'train time:{elapsed_time}')
    