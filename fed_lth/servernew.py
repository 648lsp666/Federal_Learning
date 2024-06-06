import socket,socketserver
import time
from conf import conf
from global_model import Global_model
from pruning_utils import *
import os,struct,pickle
from chooseclient import simulated_annealing
import torch
import random


  #向单个客户端发送文件函数
  # sock:接收方socket
  # filename：文件路径
def send_file( sock, filename):
  with open(filename, 'rb') as file:
    file_size = os.path.getsize(filename)
    print(f"Sending {filename} of size {file_size} bytes.")
    sock.sendall(struct.pack('>I', file_size))  # 大端序打包文件大小
    # 发送文件内容
    while True:
      chunk = file.read(1024)
      if not chunk:
        print(f"File {filename} sent successfully.")
        break
      sock.sendall(chunk)

# 发送数据函数     # sock:接收方socket
def send_data(client_socket, data):
  # 序列化数据
  data_bytes = pickle.dumps(data)
  # 发送数据大小
  data_size = len(data_bytes)
  client_socket.sendall(struct.pack('>I', data_size))
  # 发送数据内容
  client_socket.sendall(data_bytes)




# 接收数据函数     # sock:发送方socket
def recv_data(sock ,expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(msg)

	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		#print(msg)
		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	return msg

#广播函数，要发送的客户端列表clients，发送的类型type 'file' / 'data',发送的消息/文件路径 content
def broadcast( clients, content_type, content):
  # 发送文件
  if content_type=='file':
  # 遍历所有客户端，向他们发送消息/文件
    for client in clients:
      send_file(client,content)
  if content_type=='data':
    for client in clients:
      send_data(client,content) 


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

  return listening_sock,client_sock_all

# 联邦训练函数
def fed_train(clients,global_model):
	print(f'global train')
		#广播训练命令
	broadcast(clients,'data','train')
	#直接下发模型 覆盖客户端本地模型
	broadcast(clients,'data',global_model.model)
	#接受客户端的更新
	for sock in clients:
		data=recv_data(sock)			
		client_update.append(data)
	#接收到全部更新,开始聚合
	print('aggregate')
	global_model.aggregate(client_update)
	torch.save(global_model,os.path.join(conf['temp_path'],f'global{global_epoch}_model'))




if __name__=='__main__':
  # 初始化
	global_model=Global_model()
	#记录当前连接的客户端
	clients=[]
	# 全局联邦训练周期,从0开始
	global_epoch=0
	#全局模型效果
	client_acc=[]
	client_loss=[]
	# 保存客户端信息
	client_info=dict()
	#初始客户端组的id
	group_id=0
	#剪枝间隔轮数
	prune_step=5
  # 保存客户端更新
	client_update=[]
  # 保存客户端分组
	groups=[]
    



	# 建立连接
	server,clients=conn()
	# 下发id和初始模型
	for id in range(len(clients)):    
		send_data(clients[id],id)
		send_data(clients[id],global_model.model)
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
		#data[0]客户端id; data[1] 剪枝率
		client_info[data[0]]['prune_ratio']=data[1]
            
	# 开始模拟退火分组
	# groups=simulated_annealing() 
	groups=[[0]]#测试用
	group_id=0
	print('group finish')
  #分组完成
     
	# 开始联邦剪枝过程
	print('start fed prune')
	while group_id<=len(groups):
		weight_with_mask = global_model.model.state_dict()
		global_model.init_ratio()
		#测试用例Target_ratio
		target_ratio = 0.8
		while global_epoch<= conf['global_epoch']:
			#训练
			# 选择参与的客户端
			participants=[clients[i] for i in groups[group_id]]
			#确定目标剪枝率
			# target_ratio=max([client_info[id] for id in groups[group_id]])
			fed_train(participants, global_model)
			global_epoch +=1
			if global_epoch % prune_step == 0:
				#每过几轮触发一次剪枝
				print(f'Unstruct Prune, Prune Ratio: {global_model.ratio}')
				# 非结构化剪枝（可迭代）
				# global_model.u_prune(global_model.ratio)
				# weight_with_mask = global_model.model.state_dict()
				# remove_prune(global_model.model, conv1=True)
				# 如果达到目标剪枝率，跳出循环
				if global_model.ratio == target_ratio:
					print('Reach Target Ratio')
					break
		# 结构化剪枝重组
		print('Structure Prune')
		#mask_weight = global_model.regroup(weight_with_mask)
		# global_model.refill(weight_with_mask)
		# remove_prune(global_model.model, conv1=False)
		# check_sparsity(global_model.model, conv1=False)
		#重置模型参数		

		#切换客户端组	
		group_id+=1

	print('fed prune finish')


  
  