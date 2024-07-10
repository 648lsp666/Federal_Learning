import socket,socketserver
import time
from conf import conf
from global_model import Global_model
from pruning_utils import *
import os,struct,pickle
from chooseclient import client_group
import torch
import random
from gzip import compress,decompress

  #向单个客户端发送文件函数
  # sock:接收方socket
  # filename：文件路径
# def send_file( sock, filename):
#   with open(filename, 'rb') as file:
#     file_size = os.path.getsize(filename)
#     print(f"Sending {filename} of size {file_size} bytes.")
#     sock.sendall(struct.pack('>I', file_size))  # 大端序打包文件大小
#     # 发送文件内容
#     while True:
#       chunk = file.read(1024)
#       if not chunk:
#         print(f"File {filename} sent successfully.")
#         break
#       sock.sendall(chunk)

# 发送数据函数     # sock:接收方socket
def send_data(client_socket, data):
  # 序列化数据
  data_bytes = compress(pickle.dumps(data))
  # 发送数据大小
  data_size = len(data_bytes)
  client_socket.sendall(struct.pack('>I', data_size))
  # 发送数据内容
  client_socket.sendall(data_bytes)




# 接收数据函数     # sock:发送方socket
def recv_data(sock ,expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(decompress(msg))

	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		#print(msg)
		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	return msg

#广播函数，要发送的客户端列表clients，发送的类型type 'file' / 'data',发送的消息/文件路径 content
def broadcast( socks, content_type, content):
  # 发送文件
  # if content_type=='file':
  # # 遍历所有客户端，向他们发送消息/文件
  #   for sock in socks:
  #     send_file(sock,content)
	if content_type=='data':
		for sock in socks:
			send_data(sock,content) 


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
	#获取客户端id对应的socket对象，需要训练的发送train命令，不参与的发送wait
	participants=[]
	wait_client=[]
	for id in range(len(clients)):
		if id in part_id:
			participants.append(clients[id])
		else:
			wait_client.append(clients[id])
	broadcast(wait_client,'data','wait')
	#广播训练命令
	broadcast(participants,'data','train')
	#直接下发模型 覆盖客户端本地模型
	broadcast(participants,'data',global_model.model)
	#接受客户端的更新、loss、acc
	client_update=[]
	client_acc=[]
	client_loss=[]
	for sock in participants:
		data=recv_data(sock)			
		client_update.append(data[0])
		client_loss.append(data[1])
		client_acc.append(data[2])
	#接收到全部更新,开始聚合
	global_model.aggregate(client_update,client_loss,client_acc)
	print(f'global epoch {global_model.global_epoch}' )
	torch.save(global_model,os.path.join(conf['temp_path'],f'global{global_model.global_epoch}_model'))

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
    



	# 建立连接、下发id和初始化模型
	server,clients=conn()
	# 广播分组命令
	broadcast(clients,'data','group')
	# 接收客户端信息
	client_info=dict()
#单个客户端信息结构 
# info= {'id':客户端id
# 'data_dis':list,数据分布
# 'train_data_len':number, 训练集大小
# 'train_time':number, 训练时间
# 'prune_ratio':number 剪枝率，默认是0}
	for sock in clients:
		data=recv_data(sock)
    # 客户端id作为key
		client_info[data['id']]=data
		print(f'recv info client{data["id"]}')
  # 接收完客户端信息 开始客户端分组
	print('start group')
  # 将平均值作为训练时间阈值
	time_list=[client_info[id]['train_time']for id in client_info]
	avgtime=sum(time_list)/len(time_list)
	#广播时间阈值
	broadcast(clients,'data',avgtime)
  #等待客户端返回剪枝率
	for sock in clients:
		data=recv_data(sock)
		#data[0]客户端id; data[1] 参数稀疏率  data[2]通道稀疏率，tp剪枝用
		client_info[data[0]]['prune_ratio']=data[1]
		client_info[data[0]]['channel_sparsity']=data[2]
            
	# 开始模拟退火分组
	# groups=client_group(client_info=client_info) 
	groups=[[0],[1,2]]#测试用
	group_id=0
	print('group finish')
  #分组完成
	# 训练100轮作为每次重置后参数
	# rewind_weight 应当在训练前被定义
	while global_model.global_epoch<10:
		fed_train(groups[group_id], global_model)
		rewind_weight = global_model.model.state_dict()

	# 开始联邦剪枝过程
	print('start fed prune')
	while group_id<len(groups) and global_model.global_epoch<conf['global_epoch']:
		#剪枝间隔轮数
		prune_step=5
		weight_with_mask = global_model.model.state_dict()
		global_model.init_ratio()
		target_ratio = max([client_info[id]['prune_ratio'] for id in groups[group_id]])
		channel_sparsity = max([client_info[id]['channel_sparsity'] for id in groups[group_id]])
		#测试用例Target_ratio
		while global_model.global_epoch< conf['global_epoch']:
			#训练
			fed_train(groups[group_id], global_model)
			if global_model.global_epoch % prune_step == 0:
				#每过几轮触发一次剪枝
				print(f'Unstruct Prune, Prune Ratio: {global_model.ratio}')
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
		print(f'Refill Struct Prune')
		prune_mask = global_model.refill(weight_with_mask, mask_only=True)
		# Recover weight
		global_model.model.load_state_dict(rewind_weight)
		# Apply Sparity to model
		prune_model_custom(global_model.model, prune_mask, conv1=False)
		remove_prune(global_model.model, conv1=False)
		# TP Permenant Prune
		global_model.model.zero_grad()
		trace_input, _ = next(iter(global_model.train_loader))
		######################把这里的0.3替换为通道剪枝率#######################
		print('Final sparsity:' + str(100 *
									  global_model.tp_prune(trace_input.to(global_model.device),
															channel_sparsity, imp_strategy='Magnitude',
															degree=1)
									  ) + '%')
		#切换客户端组	
		group_id+=1
	print('fed prune finish')
  # 剪枝结束，微调  
	while global_model.global_epoch<conf['global_epoch']:
		group=random.sample(groups , 1)[0]
		fed_train(group, global_model)
	

  # 下发评估指令
	broadcast(clients,'data','eval')
	broadcast(clients,'data',global_model.model)
	# 微调结束，全局模型评估
	global_model.eval()
  #等待客户端返回统计数据
	eval_info=[]
	for sock in clients:
		data=recv_data(sock)
		eval_info.append(data)
	with open('result/evalinfo.pkl','wb') as f:
		pickle.dump(eval_info,f)
	with open('result/client_info.pkl','wb') as f:
		pickle.dump(client_info,f)
	with open('result/global_acc.pkl','wb') as f:
		pickle.dump(global_model.acc,f)
	print('FL done')
  