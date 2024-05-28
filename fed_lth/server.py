# @zhy
# fedlth项目的服务器模块


# 非阻塞模块
import socket,socketserver
import time
from conf import conf
from global_model import Global_model
from pruning_utils import *
import os,struct,pickle
from chooseclient import simulated_annealing
import torch

# class fed_server(socketserver.ThreadingTCPServer):
#   def __init__(self):
#     # 按照配置中的模型信息获取模型，这里使用的是torchvision的models模块内置的ResNet-18模型
#     # 模型下载后，令其作为全局初始模型
#     self.global_model = models.get_model(conf["model_name"]) 
#     # 生成一个测试集合加载器
#     # shuffle=True打乱数据集
#     # self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
#     #当前连接的客户端
#     self.clients=[]
#     # 就绪的客户端数目
#     self.ready_client=0
#     # 全局联邦训练周期
#     self.global_epoch=0
#     #全局模型效果
#     self.client_acc=[]
#     self.client_loss=[]
#     #FL服务器当前的阶段，初始化为'conn'与客户端建立连接；‘group’:客户端分组；‘prune’:训练剪枝；‘train’：剪枝完成；‘finish’训练完成
#     self.stage='conn'

  # def conn_clients(self,request):
  #   #记录客户端列表
  #   self.clients.append(request)
  #   print(f'客户端{len(self.clients)}已连接:{request}')
  #   if len(self.clients)== conf['num_client']:
  #     #所有客户端已经连接
  #     print(f'All clients connected! Clients:{len(self.clients)}')
  #     #所有客户端就绪
  #     self.ready_client=len(self.clients)   



# 定义消息处理类
class Fed_handler(socketserver.BaseRequestHandler):
	global_model=Global_model()
	#记录当前连接的客户端
	clients=[]
	# 就绪的客户端数目
	ready_client=0
	# 全局联邦训练周期,从0开始
	global_epoch=0
	#全局模型效果
	client_acc=[]
	client_loss=[]
	#FL服务器当前的阶段，初始化为'conn'与客户端建立连接；‘group’:客户端分组；‘prune’:训练剪枝；‘train’：剪枝完成；‘finish’训练完成
	stage='conn'
	client_info_list=[]
	#初始客户端组的id
	group_id=0
	#剪枝间隔轮数
	prune_step=5
	client_update=[]
	groups=[]
	

	# 首先执行setup方法，然后执行handle方法，最后执行finish方法
	# 如果handle方法报错，则会跳过
	# setup与finish无论如何都会执行
	# 一般只定义handle方法即可
	def setup(self):
		pass

	#广播函数，要发送的客户端列表clients，发送的类型type 'file' / 'data',发送的消息/文件路径 content
	def broadcast(self, clients, content_type, content):
		# 发送文件
		if content_type=='file':
		# 遍历所有客户端，向他们发送消息/文件
			for client in clients:
				self.send_file(client,content)
		if content_type=='data':
			for client in clients:
				self.send_data(client,content)    
			
		#向单个客户端发送文件函数
		# sock:接收方socket
		# filename：文件路径
	def send_file(self, sock, filename):
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
		
	#接收文件函数
	def recv_file(self,sock):
		# 接收文件大小
		size_data = sock.recv(4)
		file_size = struct.unpack('>I', size_data)[0]  # 大端序解包文件大小

		# 接收文件内容并写入文件
		with open('received_file', 'wb') as file:
				while file_size > 0:
						chunk = sock.recv(min(1024, file_size))
						file_size -= len(chunk)
						file.write(chunk)
		print("File received successfully.")

	def recv_data(self, expect_msg_type=None):
		sock=self.request
		msg_len = struct.unpack(">I", sock.recv(4))[0]
		msg = sock.recv(msg_len, socket.MSG_WAITALL)
		msg = pickle.loads(msg)

		if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
			#print(msg)
			raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
		return msg

	# 发送数据函数     
	# sock:接收方socket
	def send_data(self,client_socket, data):
		# 序列化数据
		data_bytes = pickle.dumps(data)
		# 发送数据大小
		data_size = len(data_bytes)
		client_socket.sendall(struct.pack('>I', data_size))
		# 发送数据内容
		client_socket.sendall(data_bytes)


	#与客户端建立链接，添加客户端列表self.clients，在handle中调用
	def conn_clients(self):
		#添加到客户端列表
		self.clients.append(self.request)
		print(f'客户端{len(self.clients)}已连接:{self.request}')
		#向该客户端发送id和初始模型,客户端id从0编号，len-1
		self.request.sendall(struct.pack('>I',len(self.clients)-1))#发送id
		self.send_file(self.request,conf['init_model'])#发送初始模型

		if len(self.clients)== conf['num_client']:
			#所有客户端已经连接
			print(f'All clients connected! Clients:{len(self.clients)}')
			#开始客户端分组
			self.stage='group'
			self.broadcast(self.clients,'data','group')

	#客户端分组过程
	def client_group(self):
		while self.ready_client<conf['num_client']:
			#第一步 接收客户端训练时间等信息
			# client_info=
	# {'data_dis':list,数据分布
	#                    'train_data_len':number, 训练集大小
	#                    'train_time':number, 训练时间
	#                    'prune_ratio':number 剪枝率，默认是0}
			data=self.recv_data()
			self.client_info_list.append(data)
			self.ready_client += 1
		#接收到所有客户端数据，计数重新归零
		self.ready_client=0
		# 将平均值作为训练时间阈值
		time_list=[info['train_time']for info in self.client_info_list]
		avgtime=sum(time_list)/len(time_list)
		#广播时间阈值
		self.broadcast(self.clients,'data',avgtime)
		#等待客户端返回剪枝率
		while self.ready_client<conf['num_client']:
			self.ready_client+=1
			data=self.recv_data()
			#data[0]客户端id; data[1] 剪枝率
			for info in self.client_info_list:
				if info['id']==data[0]:
					info['prune_ratio']=data[1]
		# 全部接收到消息，计数重置
		self.ready_client=0
		#设置剪枝率
		prune_ratio=[info['prune_ratio']for info in self.client_info_list]
		# avgtime=sum(time_list)/len(time_list)
		print(prune_ratio)
		self.broadcast(self.clients,'data',prune_ratio)
		# 开始模拟退火分组
		# groups,_,_=simulated_annealing() 
		
		groups=[[0]]#测试用
		#分组完成
		self.stage=='prune'
		return groups
		
	# 全局训练函数
	def train(self,groups):
		# 选择参与的客户端组
		parts=[self.clients[i] for i in groups[self.group_id]]
		#广播训练命令
		self.broadcast(parts,'data','train')
		#直接下发模型 覆盖客户端本地模型
		self.broadcast(parts,'data',self.global_model.model)
		#接受客户端的更新
		while self.ready_client<conf['num_client']:
			self.ready_client+=1
			data=self.recv_data()
			self.client_update.append(data)
		#接收到全部更新,开始聚合
		self.global_model.aggregate(self.client_update)
		self.global_epoch+=1
		torch.save(self.global_model,os.path.join(conf['temp_path'],f'global{self.global_epoch}_model'))
			



	#LTH迭代剪枝过程
	#group_id:客户端组id
	# prune_step:剪枝间隔轮数
	def fed_prune(self,groups,prune_step):
		mask_weight = self.global_model.model.state_dict()
		self.global_model.init_ratio()
		while self.global_epoch<= conf['global_epoch']:
			#训练
			self.train(groups)
			if self.global_epoch % prune_step == 0:
				#每过几轮触发一次剪枝
				print('Unstruct Prune')
				#非结构化剪枝（可迭代）
				self.global_model.u_prune(self.global_model.ratio)
				mask_weight = self.global_model.model.state_dict()
				remove_prune(self.global_model.model, conv1=True)
				#如果达到目标剪枝率，跳出循环
				self.global_model.increase_ratio(target_ratio)
				if self.global_model.ratio == target_ratio:
					print('Reach Target Ratio')
					break
		# 结构化剪枝重组
		print('Structure Prune')
		mask_weight = self.global_model.regroup(mask_weight)
		remove_prune(self.global_model.model, conv1=False)
		#重置模型参数		

		#切换客户端组	
		self.group_id+=1
		if self.group_id==len(self.groups):
			# 剪枝结束，开始微调训练
			self.stage='train'
				

	#服务器总处理响应函数，通过self.stage确定当前的执行流程
	def handle(self):
		#与客户端建立连接，记录客户端列表
		if self.stage=='conn':
			self.conn_clients()
		while True:
			if self.stage=='group':
				#客户端分组阶段
				self.client_group()
			if self.stage=='prune':
				self.fed_prune(5)
			


	def finish(self):
		pass


if __name__ == "__main__":
  # 创建多线程实例
  server = socketserver.ThreadingTCPServer((conf['ip'], conf['port']), Fed_handler)
  print('Wait for client...')
  # 开启多线程，等待连接
  server.serve_forever()