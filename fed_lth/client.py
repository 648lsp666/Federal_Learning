import socket
import time,struct,os
from conf import conf
from local_model import Local_model
import pickle
import torch 


# @zhy
# fedlth项目的客户端模块

# sock:接收方socket
# filename：保存文件的路径
def recv_file(sock,filename):
  # 接收文件大小
  size_data = sock.recv(4)
  file_size = struct.unpack('>I', size_data)[0]  # 大端序解包文件大小
  print(f'recv file size:{file_size}')
  # 接收文件内容并写入文件
  with open(filename, 'wb') as file:
    while file_size > 0:
      chunk = sock.recv(min(1024, file_size))
      file_size -= len(chunk)
      file.write(chunk)
  print("File received successfully.")


# recv:接收方socket
# filename：保存文件的路径
def send_file(sock, filename):
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

# 接收数据函数     
# sock:接收方socket
def recv_data(sock, expect_msg_type=None):
  data=sock.recv(4)
  if not data:
    return ''
  msg_len = struct.unpack(">I", data)[0]
  msg = sock.recv(msg_len, socket.MSG_WAITALL)
  msg = pickle.loads(msg)

  if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
    #print(msg)
    raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
  return msg

# 发送数据函数     
# sock:接收方socket
def send_data(client_socket, data):
    # 序列化数据
    data_bytes = pickle.dumps(data)
    # 发送数据大小
    data_size = len(data_bytes)
    comm_datasize.append(data_size)
    client_socket.sendall(struct.pack('>I', data_size))
    # 发送数据内容
    client_socket.sendall(data_bytes)




if __name__ == "__main__":
  print('Client strat. Waiting...')
  # 记录通信数据量
  comm_datasize=[]
  total_time=[]
  # 服务端为TCP方式，客户端也采用TCP方式，默认参数即为TCP
  client = socket.socket()
  # 连接主机
  client.connect((conf['ip'],conf['port']))
  #接收服务器分配的客户端id,id从0开始编号
  client_id=recv_data(client)
  print(f'Client id:{client_id}')
  #接收初始模型
  # recv_file(client, os.path.join(conf['temp_path'],f'client{client_id}_init_model'))
  model=recv_data(client)

  #初始化本地模型
  local_model=Local_model(client_id,model)

  # 等等待服务器下一步命令
  op=recv_data(client)
  # while op=='':
  #   op=recv_data(client)

  #开始客户端分组
  if op=='group':
    #第一步 测量本地训练时间
    train_time=local_model.time_test()
    #第二步 上传本地信息:客户端id 训练集大小、数据分布、训练时间
    client_info={'id':client_id,
      'data_dis':local_model.train_dis,
      'train_data_len':local_model.train_len, 
      'train_time':train_time, 
      'prune_ratio':0,
      'channel_sparsity':0 }
    print(client_info)
    send_data(client,client_info)
    #第三步 接收服务器下发的时间阈值，本地随机剪枝测时间
    train_time_T=recv_data(client)
    # 在prune_ratio函数中测出客户端在该时间阈值下的剪枝率
    prune_ratio,channel_sparsity=local_model.prune_ratio(train_time_T)
    # 上传该剪枝率
    send_data(client,[client_id,prune_ratio,channel_sparsity])
    print('group finish')

  # 等待服务器下一步命令
  op=recv_data(client)
  #本地训练
  while op=='train':
    print('local train')
    #本地训练
    #直接用全局模型覆盖本地模型
    local_model.model=recv_data(client)
    #本地训练
    train_time=local_model.local_train(local_model.train_data,conf['local_epoch'])
    total_time.append(train_time)
    # 上传状态字典
    send_data(client,local_model.model.state_dict()) 
    op=recv_data(client)
  if op=='eval':
    print('fed finish,start eval' )
    # 上传测试数据
    eval_info={'id':client_id,
               'total_time':total_time,
               'comm_size':comm_datasize,
               'loss':local_model.loss,
               'acc':local_model.acc}
    send_data(client,eval_info)
    
    

