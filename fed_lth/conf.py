# @zhy
# fedlth的设置模块
conf={
    #服务器IP地址
    "ip":"127.0.0.1",
    #服务监听端口
    "port":5000,


    # # 模型信息：即当前任务使用的模型结构，此处为ResNet-18图像分类模型
    "model_name" : "resnet18",	
    "arch_name" : "res18",
    #数据集名称
    "dataset_name":'cifar10',
    #数据集位置
    "dataset_dir":'data',
    # "dataset_dir":'datasets/cifar10',
    "use_sparse_conv" : False,

    #初始模型的文件路径：
    "pretrained" : "pretrained/1checkpoint.pth.tar",
    "init_model":'models\init_resnet18.pt',
    # 临时文件路径 存放训练过程发送/接收的数据
    "temp_path":"temp",
    #总客户端数量
    "num_client" : 1,
	

    #noniid设置： 暂未使用
    'n_class':10,
    'nsamples':20,
    'rate_unbalance':1.0,
	
    #全局迭代次数：即服务端和客户端的通信次数
    #通常会设置一个最大的全局迭代次数，但在训练过程中，只要模型满足收敛的条件，那么训练也可以提前终止
	"global_epoch" : 10,
	
    #本地模型的迭代次数：即每一个客户端在进行本地模型训练时的迭代次数
	"local_epoch" : 1,
	
    #每次选取k个客户端参与迭代
	"k" : 1,
	
    #本地模型进行训练时的参数-每个batch的大小
	"batch_size" : 32,
	
    #本地模型进行训练时的参数-学习率
	"lr" : 0.01,
    "decreasing_lr" : "80,120",
    #本地模型进行训练时的参数-momentum
	"momentum" : 0.0001,
	#weight decay
    "weight_decay" : "0.0001",

    #本地模型进行训练时的参数-正则化参数
	"lambda" : 1,

    "prune_type" : "lt",
    "start_ratio" : 0.1,

    #Global_model设备
    "global_dev" : 'cuda:0',
    'device':'cuda'
}
