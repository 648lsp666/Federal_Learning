import socket
import time,struct,os
from conf import conf
from local_model import Local_model
import pickle

# def get_extension_after_last_dot(my_string):
#     """
#     获取字符串中最后一个点之后的字符（扩展名）。

#     Args:
#         my_string (str): 输入的字符串。

#     Returns:
#         str: 返回最后一个点之后的字符（扩展名），如果没有找到点则返回空字符串。
#     """
#     last_dot_index = my_string.rfind('.')  # 查找最后一个点的索引
#     if last_dot_index != -1:  # 确保找到了点
#         return my_string[last_dot_index:]  # 返回点之后的字符2
#     else:
#         return ""  # 如果没有找到点，则返回空字符串

# @zhy
# fedlth项目的客户端模块

# sock:接收方socket
# filename：保存文件的路径
def recv_file(sock,filename):
  # 接收文件大小
  size_data = sock.recv(4)
  file_size = struct.unpack('>I', size_data)[0]  # 大端序解包文件大小

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
def recv_data(conn):
  # 接收数据大小
  size_data = conn.recv(4)
  if not size_data:
    return None
  data_size = struct.unpack('>I', size_data)[0]

  # 接收数据内容
  data = b""
  while len(data) < data_size:
    packet = conn.recv(4096)
    if not packet:
      break
    data += packet

  return pickle.loads(data)







if __name__ == "__main__":
  # 服务端为TCP方式，客户端也采用TCP方式，默认参数即为TCP
  client = socket.socket()
  # 连接主机
  client.connect((conf['ip'],conf['port']))
  #接收服务器分配的客户端id
  client_id=struct.unpack('>I', client.recv(4))[0]
  #接收初始模型
  recv_file(client, os.path.join(conf['temp_path'],f'client{client_id}_init_model'))

  #初始化本地模型
  local_model=Local_model(client_id)

  #开始客户端分组
  op=recv_data(client)
  if op=='group':
    #第一步 测量本地训练时间
    train_time=local_model.time_test()
    #第二步 上传本地信息:客户端id 训练集大小、数据分布、训练时间
    client_info={'id':client_id,
      'data_dis':local_model.train_dis,
      'train_data_len':local_model.train_len, 
      'train_time':train_time, 
      'prune_ratio':0 }
    client.sendall(pickle.dumps(client_info))
    #第三步 接收服务器下发的时间阈值，本地随机剪枝测时间
    train_time_T=recv_data(client)
    # 在prune_ratio函数中测出客户端在该时间阈值下的剪枝率
    prune_ratio=local_model.prune_ratio(train_time_T)
    # 上传该剪枝率
    client.sendall(pickle.dumps([client_id,prune_ratio]))
    



  # 定义发送循环信息，等待服务器分组和调用训练
  while True:
    # 接收主机信息 每次接收缓冲区1024个字节
    data = client.recv(1024)
    # 打印接收数据
    print(data.decode())
    
    

