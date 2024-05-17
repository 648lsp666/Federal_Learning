# @zhy 暂时没用 准备用来获取数据集和分割数据集
import torch 
import numpy as np
from torchvision import datasets, transforms

from conf import conf

# 根据客户端id获取数据集
def get_dataset(client_id):
  dir=conf['dataset_dir']
  name=conf['dataset_name']
  # download=true表示从下载数据集并把数据集放在root路径中
  if name=='mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(dir, train=True, download=True, transform=trans_mnist)
    eval_dataset = datasets.MNIST(dir, train=False, download=True, transform=trans_mnist)
    
  elif name=='cifar10':
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
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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

  # 按客户端id划分数据集 随机抽取样本（iid）
  train_range = list(range(len(train_dataset)))
  eval_range = list(range(len(eval_dataset)))
  # train_data_len是每个客户端的数据量
  train_data_len = int(len(train_dataset) / conf['num_client'])
  eval_data_len = int(len(eval_dataset) / conf['num_client'])
  # 根据客户端的id来平均划分训练集和测试集，indices为该id下的子训练集
  train_indices = train_range[id * train_data_len: (id + 1) * train_data_len]
  eval_indices = eval_range[id * eval_data_len: (id + 1) * eval_data_len]
  # 训练数据集的加载器，自动将数据分割成batch
  # sampler定义从数据集中提取样本的策略
  # 使用sampler：构造数据集的SubsetRandomSampler采样器，它会对下标进行采样
  # train_dataset父集合
  # sampler指定子集合
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=conf["batch_size"], sampler=torch.utils.data.sampler.SubsetRandomSampler(eval_indices)) 
  return train_loader, eval_loader
  

# 以下用于划分noniid数据集，暂时没用
def cifar_extr_noniid(train_dataset, test_dataset, num_users = conf['num_client'], n_class=conf['n_class'], num_samples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)} 
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])


    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)   
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test

def get_dataset_cifar10_extr_noniid(num_users = conf['num_client'], n_class=conf['n_class'], nsamples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    data_dir = conf['dataset_dir']
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

if __name__=='__main__':
   get_dataset()