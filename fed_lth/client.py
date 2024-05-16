import socket
import time,struct,os
from conf import conf

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



# recv:接收方socket
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

if __name__ == "__main__":
  
  # 服务端为TCP方式，客户端也采用TCP方式，默认参数即为TCP
  client = socket.socket()
  # 连接主机
  client.connect((conf['ip'],conf['port']))
  #接收服务器分配的客户端id
  client_id=struct.unpack('>I', client.recv(4))[0]
  #接收初始模型
  recv_file(client,os.path.join(conf['temp_path'],f'client{client_id}_init_model'))

  #开始客户端分组

  #第一步 测量本地训练时间

  # 定义发送循环信息
  while True:
    # 接收主机信息 每次接收缓冲区1024个字节
    data = client.recv(1024)
    # 打印接收数据
    print(data.decode())
    
    

