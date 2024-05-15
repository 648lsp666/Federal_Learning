import torch 
from torchvision import datasets, transforms

def get_dataset(dir, name):
  # download=true表示从下载数据集并把数据集放在root路径中
  if name=='mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(dir, train=True, download=True, transform=trans_mnist)
    eval_dataset = datasets.MNIST(dir, train=False, download=True, transform=trans_mnist)
    
  elif name=='cifar':
        # transform：图像类型的转换
        # 用Compose串联多个transform操作
    transform_train = transforms.Compose([
            # 四周填充0，图像随机裁剪成32*32
      transforms.RandomCrop(32, padding=4),
            # 图像一半概率翻转，一半概率不翻转
      transforms.RandomHorizontalFlip(),
            # 将图片(Image)转成Tensor，归一化至[0, 1]
      transforms.ToTensor(),
            # 标准化至[-1, 1]，规定均值和标准差
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
            # 将图片(Image)转成Tensor，归一化至[0, 1]
      transforms.ToTensor(),
            # 标准化至[-1, 1]，规定均值和标准差
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
        #得到训练集
    train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                    transform=transform_train)
        #得到测试集
    eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    
  # 该函数返回训练集和测试集
  return train_dataset, eval_dataset
