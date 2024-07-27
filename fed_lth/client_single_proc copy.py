import socket
import time,struct,os
from conf import conf
from local_model import Local_model
import pickle
from gzip import compress,decompress
from dataset import get_dataset,get_data_indices

# @zhy
# fedlth项目的客户端模块

# 接收数据函数     # sock:发送方socket
def recv_data(sock ,expect_msg_type=None):
  msg_len = struct.unpack(">I", sock.recv(4))[0]
  try: 
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(decompress(msg))
    sock.sendall(struct.pack('>I', 200))
    return msg
  except Exception as e:
    print("An error occurred:", e)
    # 返回错误代码
    sock.sendall(struct.pack('>I', 400))


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
    self.acc_list=[]
    self.loss_list=[]
    self.train_time_list=[]
    self.comm_datasize_list=[]
    print(f'Client id:{self.id}. Waiting...')



if __name__ == "__main__":
  train_data, test_data = get_dataset(False)
  client_list=[]
  # 单个脚本模拟多客户端
  for i in range(conf['v_client']):
    client_list.append(Fed_client())
  # 分配数据集
  train_indices, eval_indices,train_dis=get_data_indices(train_data,test_data)
  for client in client_list:
    client.train_indices, client.eval_indices,client.train_dis =train_indices[client.id], eval_indices[client.id],train_dis[client.id]
  print('Init done')
  for client in client_list:
    # 等等待服务器下一步命令
    op=recv_data(client.sock)
    #开始客户端分组
    if op=='group':
      # 接收初始模型
      model=recv_data(client.sock)
      # 创建本地模型对象
      local_model=Local_model(model)
      #第一步 测量本地训练时间
      epoch_time=local_model.time_test()
      train_len=len(client.train_indices)
      train_time = train_len/conf['batch_size']*epoch_time*conf['local_epoch']
      #第二步 上传本地信息:客户端id 训练集大小、数据分布、训练时间
      client_info={'id':client.id,
        'data_dis':client.train_dis,
        'train_data_len':train_len, 
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
    prune_ratio,channel_sparsity=local_model.prune_ratio(train_time_T)
    # 上传该剪枝率
    send_data(client.sock,[client.id,prune_ratio,channel_sparsity])
  print('group finish')

  # 训练阶段，等待服务器下一步命令
  global_epoch = 0
  while global_epoch < conf['global_epoch']:
    for client in client_list: 
      op=recv_data(client.sock)
      if op=='train':
        global_epoch=recv_data(client.sock)
        print(f'global epoch {global_epoch},client {client.id} local train')
        #本地训练
        #接收全局模型参数
        model_para=recv_data(client.sock)
        comm_size=len(compress(pickle.dumps(model_para)))
        client.comm_datasize_list.append(comm_size)
        #本地训练
        local_model.model.load_state_dict(model_para)
        train_time,weight=local_model.train(train_data,client.train_indices)
        loss,acc=local_model.eval(test_data,client.eval_indices)
        client.train_time_list.append(train_time)
        client.loss_list.append(loss)
        client.acc_list.append(acc)
        # 上传状态字典、loss 和 test acc
        send_data(client.sock,[weight,loss, acc]) 
        comm_size=len(compress(pickle.dumps(weight)))
        client.comm_datasize_list.append(comm_size)
      elif op=='wait':
        global_epoch=recv_data(client.sock)
        print(f'client {client.id} wait')
      # 全局模型剪枝后 需要重新分发模型结构
      elif op=='model':
        model=recv_data(client.sock)
        local_model=Local_model(model)
      else :
        raise Exception('OP ERROR!')
    if op=='eval':
      print('fed train finish,start eval')
      break
    if op=='wait' or op == 'train':
      global_epoch+=1
    
  
  for client in client_list:
    op=recv_data(client.sock)
    if op=='eval':
      model=recv_data(client.sock)
      local_model=Local_model(model)
      loss,acc=local_model.eval(test_data,client.eval_indices)
      print(f'client{client.id} final Acc:{acc}')
      # 上传测试数据
      eval_info={'id':client.id,
                'train_time_list':client.train_time_list,
                'comm_size_list':client.comm_datasize_list,
                'loss_curve':client.loss_list,
                'acc_curve':client.acc_list,
                'final_acc':acc,
                'final_loss':loss}
      send_data(client.sock,eval_info)

  print('FL done')
