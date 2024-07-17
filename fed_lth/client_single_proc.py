import socket
import time,struct,os
from conf import conf
from local_model import Local_model
import pickle
from gzip import compress,decompress

# @zhy
# fedlth项目的客户端模块

# 接收数据函数     # sock:发送方socket
def recv_data(sock ,expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(decompress(msg))
	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		#print(msg)
		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	# 表示成功接收
	sock.sendall(struct.pack('>I', 200))
	return msg

# 发送数据函数     # sock:接收方socket
def send_data(sock, data):
	# 序列化数据
	data_bytes = compress(pickle.dumps(data))
	# 发送数据大小
	data_size = len(data_bytes)
	sock.sendall(struct.pack('>I', data_size))
	# 发送数据内容
	sock.sendall(data_bytes)
	# 接收状态码，确定是否成功
	code = struct.unpack(">I", sock.recv(4))[0]
	if code!=200:
		raise Exception(f"send data error,socket: {sock}")

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
  for i in range(10):
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
    else:
      print(op)

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
  # for client in client_list:
  #   op=recv_data(client.sock)
  #   print(op)
  print('FL done')



    
    

