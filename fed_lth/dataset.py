import os
import torch
import numpy as np
import pickle
from collections import Counter
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from conf import conf

# 获取数据集，如果是服务器直接返回loader
def get_dataset(ifserver):
    dir=conf['dataset_dir']
    name=conf['dataset_name']
        # download=true表示从下载数据集并把数据集放在root路径中
    if name=='mnist':
        trans_mnist = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
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
    else:
        print('Wrong dataset name')

    #如果是服务器 直接返回测试集用来测试全局模型的精度
    if ifserver:
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"])
        eval_loader=torch.utils.data.DataLoader(eval_dataset, batch_size=conf["batch_size"])
        return train_loader,eval_loader
    else:
        return train_dataset,eval_dataset

#返回所有客户端的训练/测试样本列表和数据分布    
def get_data_indices(train_dataset,eval_dataset):
    train_indices_list=[]
    eval_indices_list=[]
    if conf['iid']==True:
        # 按客户端id划分数据集 随机抽取样本（iid）
        train_range = list(range(len(train_dataset)))
        eval_range = list(range(len(eval_dataset)))
        # train_data_len是每个客户端的数据量
        train_data_len = int(len(train_dataset) / conf['num_client'])
        eval_data_len = int(len(eval_dataset) / conf['num_client'])
        # 根据客户端的id来平均划分训练集和测试集，indices为该id下的子训练集
        for id in range(conf['num_client']):
            train_indices_list.append(train_range[id * train_data_len: (id + 1) * train_data_len])
            eval_indices_list.append(eval_range[id * eval_data_len: (id + 1) * eval_data_len]) 

        
    else:
    #    noniid用main中生成好的数据idx生成数据集
        with open('noniid/cifar10_train.pkl','rb') as f:
            train_indices_list=pickle.load(f)
        with open('noniid/cifar10_test.pkl','rb') as f:
            eval_indices_list=pickle.load(f)

    #统计各类数据数量分布的函数，得到一个list
    train_data_dis=[]
    for indices in train_indices_list:
        subdata = Subset(train_dataset,indices)
        train_data_dis.append(dis_total(subdata))
    del subdata
    return train_indices_list, eval_indices_list,train_data_dis


#从dataloader统计数据分布,num_class是数据集总类别数量
def dis_total(dataset,num_class=conf['num_class']):
    datasize=len(dataset)
    loader=DataLoader(dataset,batch_size=128,shuffle=False)
    label_counter=Counter()
    for _, labels in loader:
    # 更新计数器
        label_counter.update(labels.tolist())  # 假设 labels 是一个 torch 张量  
    label_list=[0]*num_class
    for label, count in label_counter.items():
        label_list[label]=count/datasize
    return label_list
   

# 以下用于划分noniid数据集
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
        dict_users_train[i]=dict_users_train[i].astype(int)
        
        dict_users_test[i]=dict_users_test[i].astype(int)
        

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


# MNIST 10 classes 70000 pics, 60000 for training and 10000 for test
def mnist_extr_noniid(train_dataset, test_dataset, num_users=conf['num_client'], n_class=conf['n_class'],
                      num_samples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    num_shards_train, num_imgs_train = int(60000 / num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]), axis=0)
        dict_users_train[i] = dict_users_train[i].astype(int)
        dict_users_test[i] = dict_users_test[i].astype(int)

    return dict_users_train, dict_users_test

def get_dataset_mnist_extr_noniid(num_users=conf['num_client'], n_class=conf['n_class'], nsamples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    data_dir = conf['dataset_dir']
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,transform=apply_transform)
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

#    # dataset for global_model
# def _getdatatransformswm():
#     transform_wm = transforms.Compose([
#         transforms.CenterCrop(32),
#         transforms.ToTensor(),
#         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     return transform_wm


# def getwmloader(wm_path='data/trigger_set', batch_size=1, labels_path='labels-cifar.txt'):
#     transform_wm = _getdatatransformswm()
#     # load watermark images
#     wmloader = None

#     wmset = ImageFolderCustomClass(
#         wm_path,
#         transform_wm)
#     img_nlbl = []
#     wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
#     for idx, (path, target) in enumerate(wmset.imgs):
#         img_nlbl.append((path, int(wm_targets[idx])))
#     wmset.imgs = img_nlbl

#     wmloader = torch.utils.data.DataLoader(
#         wmset, batch_size=batch_size, shuffle=True,
#         num_workers=4, pin_memory=True)

#     return wmloader


# class CIFAR10_with_index(CIFAR10):
#     def __init__(self, root, train=True, transform=None, target_transform=None,
#                 download=False):
#         super().__init__(root, train, transform, target_transform,
#                         download)

#     def __getitem__(self, index):
#         sample = super().__getitem__(index)
#         return index, sample[0], sample[1]


# def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10'):

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#     val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True),
#                     list(range(45000, 50000)))
#     test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
#                                 pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader


# def cifar10_with_trigger_dataloaders(batch_size=128, data_dir='datasets/cifar10'):

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#     val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True),
#                     list(range(45000, 50000)))
#     test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
#     train_loader = DataLoader(train_set, batch_size=batch_size - 2, shuffle=True, num_workers=2, drop_last=False,
#                                 pin_memory=True)
#     trigger_set_loader = getwmloader()
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader, trigger_set_loader


# def cifar10_subset_dataloaders(batch_size=128, data_dir='datasets/cifar10'):

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(4500)))
#     val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True),
#                     list(range(45000, 50000)))
#     test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
#                                 pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader


# def cifar100_with_trigger_dataloaders(batch_size=128, data_dir='datasets/cifar100'):

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#     val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True),
#                     list(range(45000, 50000)))
#     test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
#                                 pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     trigger_set_loader = getwmloader()

#     return train_loader, val_loader, test_loader, trigger_set_loader


# def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100'):

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     # train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#     train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
#     # val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
#     test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
#     val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
#                                 pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader


# def fashionmnist_dataloaders(batch_size=64, data_dir='datasets/fashionmnist'):

#     train_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = Subset(FashionMNIST(data_dir, train=True, transform=train_transform, download=True),
#                         list(range(55000)))
#     val_set = Subset(FashionMNIST(data_dir, train=True, transform=test_transform, download=True),
#                     list(range(55000, 60000)))
#     test_set = FashionMNIST(data_dir, train=False, transform=test_transform, download=True)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
#                                 pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader


# def tiny_imagenet_dataloaders(batch_size=64, data_dir='datasets/tiny-imagenet-200', dataset=False, split_file=None):

#     train_transform = transforms.Compose([
#         transforms.RandomRotation(20),
#         # transforms.RandomCrop(64, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_path = os.path.join(data_dir, 'train')
#     val_path = os.path.join(data_dir, 'val')

#     if not split_file:
#         split_file = 'npy_files/tiny-imagenet-train-val.npy'
#     split_permutation = list(np.load(split_file))

#     train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
#     val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
#     test_set = ImageFolder(val_path, transform=test_transform)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     if dataset:
#         print('return train dataset')
#         train_dataset = ImageFolder(train_path, transform=train_transform)
#         return train_dataset, val_loader, test_loader
#     else:
#         return train_loader, val_loader, test_loader

if __name__=='__main__':
#    生成客户端noniid数据分布，共100客户端，每个客户端2个类别共400个样本。
    # _,_,user_groups_train, user_groups_test=get_dataset_cifar10_extr_noniid(num_users =100, n_class=2, nsamples=200, rate_unbalance=1)
    # with open('noniid\cifar10_train.pkl','wb') as f:
    #   pickle.dump(user_groups_train,f)
    # with open('noniid\cifar10_test.pkl','wb') as f:
    #   pickle.dump(user_groups_test,f)
    get_dataset(0)

    # train_dataset, test_dataset = get_dataset_femnist_noniid(3)
    # train_loader = DataLoader(train_dataset, batch_size=4)
    # for batch in train_loader:
    #     x, y = batch
    #     print(f'x: {x.shape}, y: {y.shape}')
    #     break

