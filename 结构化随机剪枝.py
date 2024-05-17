import torch
from torchvision.models import resnet18
import torch_pruning as tp
import torch.nn as nn
import torch.optim as optim

model = resnet18(pretrained=True)
example_inputs = torch.randn(1, 3, 224, 224)

class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                importance_scores = torch.abs(layer.weight.data)  # 使用 L1 范数计算重要性
                group_imp.append(importance_scores)
        if len(group_imp) == 0: return None
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp

class MySlimmingPruner(tp.pruner.MetaPruner):
    def regularize(self, model, reg):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
                m.weight.grad.data.add_(reg * torch.sign(m.weight.data))

def s_prune(ratio):
    # 使用自定义的重要性评估
    imp = MySlimmingImportance()

    # 忽略最后的分类层
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    # 初始化剪枝器
    iterative_steps = 1
    pruner = MySlimmingPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=ratio,
        ignored_layers=ignored_layers,
    )

    # 定义损失函数和优化器（可更改）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 模拟训练数据
    inputs = torch.randn(16, 3, 224, 224)  # 16张 224x224 的图片
    labels = torch.randint(0, 1000, (16,))  # 16个标签，范围在0-999之间

    # 进行稀疏训练
    for epoch in range(5):  # 训练5个epoch
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        pruner.regularize(model, reg=1e-5)  # 稀疏化
        optimizer.step()

    pruner.step()

if __name__ == '__main__':
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # 执行剪枝
    s_prune(0.5)
    # 剪枝后计算模型参数和计算量
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("参数数量: {:.2f} M => {:.2f} M".format(base_nparams / 1e6, nparams / 1e6))
    print("计算量: {:.2f} G => {:.2f} G".format(base_macs / 1e9, macs / 1e9))
