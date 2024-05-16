# 非阻塞模块
import socketserver
import time
from conf import conf

import os,struct
# python对象传输 序列化
import pickle,zlib
import models,torch,random


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
class fed_handler(socketserver.BaseRequestHandler):
 
  #记录当前连接的客户端
  clients=[]
  # 就绪的客户端数目
  ready_client=0
  # 全局联邦训练周期
  global_epoch=0
  #全局模型效果
  client_acc=[]
  client_loss=[]
  #FL服务器当前的阶段，初始化为'conn'与客户端建立连接；‘group’:客户端分组；‘prune’:训练剪枝；‘train’：剪枝完成；‘finish’训练完成
  stage='conn'


  # 首先执行setup方法，然后执行handle方法，最后执行finish方法
  # 如果handle方法报错，则会跳过
  # setup与finish无论如何都会执行
  # 一般只定义handle方法即可
  def setup(self):
    pass


  #与客户端建立链接，添加客户端列表self.clients，在handle中调用
  def conn_clients(self):
    #添加到客户端列表
    self.clients.append(self.request)
    print(f'客户端{len(self.clients)}已连接:{self.request}')
    #向该客户端发送id和初始模型
    self.request.sendall(struct.pack('>I',len(self.clients)))#发送id
    self.send_file(self.request,conf['init_model'])#发送初始模型

    if len(self.clients)== conf['num_client']:
      #所有客户端已经连接
      print(f'All clients connected! Clients:{len(self.clients)}')
      #所有客户端就绪
      self.ready_client=len(self.clients) 
      #开始客户端分组
      self.stage='group'
      self.broadcast(self.clients,'msg','group')
  
  #客户端分组过程
  def client_group(self):
    pass



  #广播函数，要发送的客户端列表clients，发送的类型type：'msg'or 'file',发送的消息/文件路径 content
  def broadcast(self, clients, content_type, content):
    # 发送消息
    if content_type=='msg':
      for client in clients:
        client.sendall(content.encode())
    # 发送文件
    if content_type=='file':
    # 遍历所有客户端，向他们发送消息/文件
      for client in clients:
        self.send_file(client,content)
      
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

  #服务器总处理响应函数，通过self.stage确定当前的执行流程
  def handle(self):
    #与客户端建立连接，记录客户端列表
    if self.stage=='conn':
      self.conn_clients()
    while True:
      if self.stage=='group':
        #客户端分组阶段
        self.client_group()
      

  def finish(self):
        pass


if __name__ == "__main__":
    # 初始化参数
    # 创建多线程实例
    server = socketserver.ThreadingTCPServer((conf['ip'], conf['port']), fed_handler)
    print('Wait for client...')
    # 开启多线程，等待连接
    server.serve_forever()