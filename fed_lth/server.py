# 非阻塞模块
import socketserver
import time
from conf import conf

# python对象传输 序列化
import pickle,zlib
import models,torch,random
import time


class fed_server(socketserver.ThreadingTCPServer):
  def __init__(self,super):
    # 按照配置中的模型信息获取模型，这里使用的是torchvision的models模块内置的ResNet-18模型
    # 模型下载后，令其作为全局初始模型
    self.global_model = models.get_model(conf["model_name"]) 
    # 生成一个测试集合加载器
    # shuffle=True打乱数据集
    # self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
    #当前连接的客户端
    self.clients=[]
    # 就绪的客户端数目
    self.ready_client=0
    # 全局联邦训练周期
    self.global_epoch=0
    #全局模型效果
    self.client_acc=[]
    self.client_loss=[]
    #FL服务器当前的阶段，初始化为'conn'与客户端建立连接；‘group’:客户端分组；‘prune’:训练剪枝；‘train’：剪枝完成；‘finish’训练完成
    self.stage='conn'

  def conn_clients(self,request):
    #记录客户端列表
    self.clients.append(request)
    print(f'客户端{len(self.clients)}已连接:{request}')
    if len(self.clients)== conf['num_client']:
      #所有客户端已经连接
      print(f'All clients connected! Clients:{len(self.clients)}')
      #所有客户端就绪
      self.ready_client=len(self.clients)   

# 定义消息处理类
class fed_handler(socketserver.BaseRequestHandler):

  global_model = models.get_model(conf["model_name"]) 
  # 生成一个测试集合加载器
  # shuffle=True打乱数据集
  # self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
  #当前连接的客户端
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


  #与客户端建立链接，添加客户端列表self.clients
  def conn_clients(self):
    self.clients.append(self.request)
    print(f'客户端{len(self.clients)}已连接:{self.request}')
    if len(self.clients)== conf['num_client']:
      #所有客户端已经连接
      print(f'All clients connected! Clients:{len(self.clients)}')
      #所有客户端就绪
      self.ready_client=len(self.clients) 
      #开始客户端分组
      self.stage='group'
      msg='group'
      self.broadcast(self.clients,msg)

  #广播函数
  def broadcast_to_all(self, clients, message):
    # 遍历所有客户端，向他们发送消息
    for client in clients:
      try:
        client.sendall(message)
      except Exception as e:
        print(f"Error broadcasting to client {client.getpeername()}: {e}")
        self.clients.remove(client)



  #服务器总处理响应函数，通过self.stage确定当前的执行流程
  def handle(self):
    #与客户端建立连接，记录客户端列表
    if self.stage=='conn':
      self.conn_clients()
      # self.clients.append(self.request)
      # print(f'客户端{len(self.clients)}已连接:{self.request}')
      # if len(self.clients)== conf['num_client']:
      #   #所有客户端已经连接
      #   print(f'All clients connected! Clients:{len(self.clients)}')
      #   #所有客户端就绪
      #   self.ready_client=len(self.clients) 
      #   #开始客户端分组
      #   self.stage='group'

        # 发送消息定义
        # msg = "已连接服务器!"
        # # 发送消息
        # conn.send(msg.encode())
        # # 进入循环，不断接收客户端消息
        # while True:
        #     # 接收客户端消息
        #     op = conn.recv(1024)
        #     if op == b'1':
        #         data = conn.recv(1024)
        #         # 打印消息
        #         print(data.decode())
        #         conn.send('接收成功'.encode())
        #     elif op == b'2':
        #         flie_ = conn.recv(1024)
        #         flie_ = flie_.decode()
        #         flie_name = str(time.time()) + flie_
        #         with open(flie_name, 'wb') as f:
        #             f.write(b'')
        #         while True:
        #             data = conn.recv(1024)
        #             if data == b'quit':
        #                 break
        #             # 写入文件
        #             with open(flie_name, 'ab') as f:
        #                 f.write(data)
        #                 # 接受完成标志
        #                 conn.send('success'.encode())
        #         print("文件接收完成")
        #         conn.send('文件接收成功'.encode())
        #     elif op == b'3':
        #         break
        #     else:
        #         conn.send('非法选项'.encode())
        # conn.close()

  def finish(self):
        pass


if __name__ == "__main__":
    # 初始化参数
    # 创建多线程实例
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 8888), fed_handler)
    print('Wait for client...')
    # 开启多线程，等待连接
    server.serve_forever()