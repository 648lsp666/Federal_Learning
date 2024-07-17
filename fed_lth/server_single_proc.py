import socket
import time
from conf import conf
from global_model import Global_model
from models.cnn_cifar import CNNCifar
from pruning_utils import *
import os,struct,pickle
from chooseclient import client_group
import torch
import random
from gzip import compress,decompress
import copy
# 发送数据函数     # sock:接收方socket
def send_data(sock, data):
	# 序列化数据
	data_bytes = compress(pickle.dumps(data))
	# 发送数据大小
	data_size = len(data_bytes)
	sock.sendall(struct.pack('>I', data_size))
	# 发送数据内容
	sock.sendall(data_bytes)
	# # 接收状态码，确定是否成功
	code = struct.unpack(">I", sock.recv(4))[0]
	if code!=200:
		raise Exception(f"send data error,socket: {sock}")

# 接收数据函数     # sock:发送方socket
def recv_data(sock ,expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(decompress(msg))
	sock.sendall(struct.pack('>I', 200))
	# 表示成功接收
	return msg


#建立连接
def conn():
	listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	listening_sock.bind((conf['ip'], conf['port']))
	client_sock_all = []

	# Establish connections to each client, up to n_nodes clients
	while len(client_sock_all) < conf['num_client']:
		listening_sock.listen(5)
		print("Waiting for incoming connections...")
		(client_sock, (ip, port)) = listening_sock.accept()
		print('Got connection from ', (ip, port))
		print(client_sock)
		client_sock_all.append(client_sock)
		# 下发id和初始模型 , id 从0开始
		id= len(client_sock_all)-1
		send_data(client_sock,id)
		send_data(client_sock,global_model.model)
	return listening_sock,client_sock_all

# 联邦训练函数
def fed_train(part_id,global_model):
	if global_model.global_epoch>=conf['global_epoch']:
		# 如果global_epoch达到直接返回
		return
	#获取客户端id对应的socket对象，需要训练的发送train命令，不参与的发送wait
	client_update=[]
	client_acc=[]
	client_loss=[]


	for id in range(len(clients)):
		sock=clients[id]
		if id in part_id:
			# 该客户端参与训练
			# 下发训练命令
			send_data(sock,'train')
			# 下发全局模型
			send_data(sock,global_model.model)
			# 接收更新信息
			data=recv_data(sock)			
			client_update.append(copy.deepcopy(data[0]))
			client_loss.append(data[1])
			client_acc.append(data[2])
		else:
			# 该客户端等待
			send_data(sock,'wait')		
	#接收到全部更新,开始聚合
	global_model.aggregate(client_update,client_loss,client_acc)
	print(f'global epoch {global_model.global_epoch}' )
	# torch.save(global_model,os.path.join(conf['temp_path'],f'global{global_model.global_epoch}_model'))

if __name__=='__main__':
  # 初始化
	global_model=Global_model()
	#记录当前连接的客户端
	clients=[]

	# 保存客户端信息
	client_info=dict()
	#初始客户端组的id
	group_id=0

  # 保存客户端更新
	client_update=[]
  # 保存客户端分组
	groups=[]
	#记录发生剪枝的轮数
	prune_round=[]
    



	# 建立连接、下发id和初始化模型
	server,clients=conn()
	print('all client connected')
	# 分组过程
	for client in clients:
		# 发送分组命令
		send_data(client,'group')
		# 接收客户端信息
		data=recv_data(client)
		client_info[data['id']]=data
		print(f'recv info client{data["id"]}')
#单个客户端信息结构 
# info= {'id':客户端id
# 'data_dis':list,数据分布
# 'train_data_len':number, 训练集大小
# 'train_time':number, 训练时间
# 'prune_ratio':number 剪枝率，默认是0}
  # 将平均值作为训练时间阈值
	time_list=[client_info[id]['train_time']for id in client_info]
	avgtime=sum(time_list)/len(time_list)
	print(f'Avgtime:{avgtime}')
	#广播时间阈值
  #等待客户端返回剪枝率
	for sock in clients:
		send_data(sock,avgtime)
		data=recv_data(sock)
		#data[0]客户端id; data[1] 参数稀疏率  data[2]通道稀疏率，tp剪枝用
		client_info[data[0]]['prune_ratio']=data[1]
		client_info[data[0]]['channel_sparsity']=data[2]
		print(f'recv client{data[0]},prune_ratio:{data[1]}')

	# # 保存客户端信息
	# with open('result/client_info10.pkl','wb') as f:
	# 	pickle.dump(client_info,f)   
	# 测试代码
	with open('result/client_info10.pkl','rb') as f:
		client_info=pickle.load(f)    
	# 开始模拟退火分组
		# 接收完客户端信息 开始客户端分组
	# client_info={
	# 	0:  {'id': 0, 'data_dis': [0.1024, 0.106, 0.0956, 0.1004, 0.0908, 0.0984, 0.1024, 0.1044, 0.0996, 0.1], 'train_data_len': 2500, 'train_time': 12.279283255338669, 'prune_ratio': 0.6179542324821345, 'channel_sparsity': 0.15}
	# }
	print('start group')
	groups=client_group(client_info=client_info) 
	# groups=[[0]]#测试用
	group_id=0
	print('group finish')

	#创建早期重置点rewind_weight 应当在训练前被定义
	while True:
		fed_train(range(len(clients)), global_model)
		rewind_weight = global_model.model.state_dict()
		if global_model.acc[-1]>0.2:
			break

	# 开始联邦剪枝过程
	print('start fed prune')
	while group_id<len(groups) and global_model.global_epoch<conf['global_epoch']:
		#分组完成
		# 精度50后开始剪枝
		while global_model.acc[-1]>conf['start_prune'] and global_model.global_epoch<conf['global_epoch']:
			fed_train(groups[group_id], global_model)
		#剪枝间隔轮数
		prune_step=2
		weight_with_mask = global_model.model.state_dict()
		global_model.init_ratio()
		target_ratio = max([client_info[id]['prune_ratio'] for id in groups[group_id]])
		channel_sparsity = max([client_info[id]['channel_sparsity'] for id in groups[group_id]])
		#测试用例Target_ratio
		while global_model.global_epoch< conf['global_epoch']:
			#训练
			fed_train(groups[group_id], global_model)
			#每过prune_step轮触发一次剪枝
			if global_model.global_epoch % prune_step == 0:
				if global_model.acc[-1]<conf['start_prune']:
					print('skip un_prune')
					continue
				
				print(f'Unstruct Prune, Prune Ratio: {global_model.ratio},Target Ratio:{target_ratio}')
				u_pruned_flag=True
				prune_round.append(global_model.global_epoch)
				# 非结构化剪枝（可迭代）
				global_model.u_prune(global_model.ratio)
				weight_with_mask = global_model.model.state_dict()
				remove_prune(global_model.model, conv1=False)
				# 如果达到目标剪枝率，跳出循环
				if global_model.ratio >= target_ratio:
					print('Reach Target Ratio')
					break
				global_model.increase_ratio(target_ratio)
			#剪枝后的一轮，如果精度下降超过5%则增大剪枝间隔
			if global_model.global_epoch % prune_step == 1:
				if global_model.acc[-1]-global_model.acc[-2]>0.05:
					prune_step+=1
		# 结构化剪枝重组
		# 如果经过非结构化剪枝才可以refill
		if u_pruned_flag==False:
			continue
		
		print(f'Refill Struct Prune')
		prune_mask = global_model.refill(weight_with_mask, mask_only=True)
		# Recover weight
		global_model.model.load_state_dict(rewind_weight)
		# Apply Sparity to model
		prune_model_custom(global_model.model, prune_mask, conv1=False)
		remove_prune(global_model.model, conv1=False)
		u_pruned_flag=False
		# TP Permenant Prune
		global_model.model.zero_grad()
		trace_input, _ = next(iter(global_model.train_loader))
		######################把这里的0.3替换为通道剪枝率#######################
		print('Final sparsity:' + str(100 *
									  global_model.tp_prune(trace_input.to(global_model.device),
															channel_sparsity, imp_strategy='Magnitude',
															degree=1)
									  ) + '%')
		torch.save(global_model.model,f'temp/CNN_group{global_model.global_epoch}_acc{global_model.acc[-1]}.pth')
		#切换客户端组	
		group_id+=1
	print('fed prune finish')
  # 剪枝结束，微调  
	while global_model.global_epoch<conf['global_epoch']:
		group=random.sample(groups , 1)[0]
		fed_train(group, global_model)

	# 微调结束，全局模型评估
	global_model.eval()

	eval_info=[]
	for sock in clients:
		# 下发指令
		send_data(sock,'eval')
		send_data(sock,global_model.model)
  #等待客户端返回统计数据
		data=recv_data(sock)
		eval_info.append(data)
	with open('result/evalinfo.pkl','wb') as f:
		pickle.dump(eval_info,f)
	with open('result/client_info.pkl','wb') as f:
		pickle.dump(client_info,f)
	with open('result/global_acc.pkl','wb') as f:
		pickle.dump(global_model.acc,f)
	with open('result/prune_round.pkl','wb') as f:
		pickle.dump(global_model.acc,f)

	print('FL done')
  