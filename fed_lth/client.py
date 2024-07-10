import socket
import time,struct,os
from conf import conf
from local_model import Local_model
import pickle
from gzip import compress,decompress

# @zhy
# fedlth项目的客户端模块

# sock:接收方socket
# filename：保存文件的路径
# def recv_file(sock,filename):
#   # 接收文件大小
#   size_data = sock.recv(4)
#   file_size = struct.unpack('>I', size_data)[0]  # 大端序解包文件大小
#   print(f'recv file size:{file_size}')
#   # 接收文件内容并写入文件
#   with open(filename, 'wb') as file:
#     while file_size > 0:
#       chunk = sock.recv(min(1024, file_size))
#       file_size -= len(chunk)
#       file.write(chunk)
#   print("File received successfully.")


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
  msg = pickle.loads(decompress(msg))

  if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
    #print(msg)
    raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
  return msg

# 发送数据函数     
# sock:接收方socket
def send_data(client_socket, data):
    # 序列化数据
    data_bytes = compress(pickle.dumps(data))
    # 发送数据大小
    data_size = len(data_bytes)
    comm_datasize.append(data_size)
    client_socket.sendall(struct.pack('>I', data_size))
    # 发送数据内容
    client_socket.sendall(data_bytes)

class Fed_client:
  def __init__(self):
      #建立连接
  # 服务端为TCP方式，客户端也采用TCP方式，默认参数即为TCP
    self.sock= socket.socket()
    # 连接主机
    self.sock.connect((conf['ip'],conf['port']))
    #接收服务器分配的客户端id,id从0开始编号
    self.id=recv_data(self.sock)
    model=recv_data(self.sock)
    #初始化本地模型
    self.local_model=Local_model(self.id,model)
    print(f'Client id:{self.id}. Waiting...')



if __name__ == "__main__":
  #该脚本单线程模拟10个客户端
  # a=Fed_client()
  client_list=[]
  for i in range(3):
    client_list.append(Fed_client())
  # 记录通信数据量
  comm_datasize=[]
  total_time=[]

  print('Init done')
  for client in client_list:
    # 等等待服务器下一步命令
    op=recv_data(client.sock)
    #开始客户端分组
    if op=='group':
      #第一步 测量本地训练时间
      train_time=client.local_model.time_test()
      #第二步 上传本地信息:客户端id 训练集大小、数据分布、训练时间
      client_info={'id':client.id,
        'data_dis':client.local_model.train_dis,
        'train_data_len':client.local_model.train_len, 
        'train_time':train_time, 
        'prune_ratio':0,
        'channel_sparsity':0 }
      print(client_info)
      send_data(client.sock,client_info)

  for client in client_list:  
    #第三步 接收服务器下发的时间阈值，本地随机剪枝测时间
    train_time_T=recv_data(client.sock)
    # 在prune_ratio函数中测出客户端在该时间阈值下的剪枝率
    prune_ratio,channel_sparsity=client.local_model.prune_ratio(train_time_T)
    # 上传该剪枝率
    send_data(client.sock,[client.id,prune_ratio,channel_sparsity])
    print('group finish')

  # 训练阶段，等待服务器下一步命令
  for global_epoch in range(conf['global_epoch']):
    print(f'global epoch {global_epoch}')
    for client in client_list: 
      op=recv_data(client.sock)
      if op=='train':
        print(f'client {client.id} local train')
        #本地训练
        #直接用全局模型覆盖本地模型
        client.local_model.model=recv_data(client.sock)
        #本地训练
        train_time,loss,acc=client.local_model.local_train(client.local_model.train_data,conf['local_epoch'])
        total_time.append(train_time)
        # 上传状态字典、loss 和 test acc
        send_data(client.sock,[client.local_model.model.state_dict(),loss, acc]) 
      elif op=='wait':
        print(f'client {client.id} wait')
      elif op=='eval':
        break
      else :
        raise Exception('ERROR!')
    if op=='eval':
      break

  for client in client_list:
    op=recv_data(client.sock)
    if op=='eval':
      print('fed finish,start eval' )
      client.local_model.model=recv_data(client.sock)
      final_acc=client.local_model.eval()
      # print(f'client{client.id} final Acc:{final_acc}')
      # 上传测试数据
      eval_info={'id':client.id,
                'total_time':total_time,
                'comm_size':comm_datasize,
                'loss_curve':client.local_model.loss,
                'acc_curve':client.local_model.acc,
                'final_acc':final_acc}
      send_data(client.sock,eval_info)
  for client in client_list:
    op=recv_data(client.sock)
    print(op)
  print('FL done')



    
    

