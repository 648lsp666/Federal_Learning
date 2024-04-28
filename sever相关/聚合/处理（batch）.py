import torch
import torchvision.models as models
import time

# 创建两个ResNet-18模型实例并加载state_dict
model1 = models.resnet18()
model2 = models.resnet18()
state_dict1 = torch.load('resnet18-5c106cde.pth')
state_dict2 = torch.load('resnet18-5c106cde.pth')
model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)

# 将模型移到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)

# 创建一个测试输入（batch_size = 1）
input_data = torch.randn(1, 3, 224, 224).to(device)

# 初始化时间测量
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

# 开始计时
start_time.record()

# 对两个模型的参数求平均（假设batch_size = 1）
avg_state_dict = {}
for key in model1.state_dict():
    avg_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2

# 创建新模型并加载平均参数
avg_model = models.resnet18()
avg_model.load_state_dict(avg_state_dict)
avg_model.to(device)

# 在batch上执行模型推理（假设batch_size = 1）
output = avg_model(input_data)

# 结束计时
end_time.record()
torch.cuda.synchronize()  # 等待GPU操作完成

# 计算时间
elapsed_time_ms = start_time.elapsed_time(end_time)
print("平均模型参数计算和推理时间：{} 毫秒".format(elapsed_time_ms))

