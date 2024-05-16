import socket
import time


def get_extension_after_last_dot(my_string):
    """
    获取字符串中最后一个点之后的字符（扩展名）。

    Args:
        my_string (str): 输入的字符串。

    Returns:
        str: 返回最后一个点之后的字符（扩展名），如果没有找到点则返回空字符串。
    """
    last_dot_index = my_string.rfind('.')  # 查找最后一个点的索引
    if last_dot_index != -1:  # 确保找到了点
        return my_string[last_dot_index:]  # 返回点之后的字符2
    else:
        return ""  # 如果没有找到点，则返回空字符串

# 服务端为TCP方式，客户端也采用TCP方式，默认参数即为TCP
client = socket.socket()
# 访问服务器的IP和端口
ip_port = ('127.0.0.1', 8888)
# 连接主机
client.connect(ip_port)
# 定义发送循环信息
while True:
    # 接收主机信息 每次接收缓冲区1024个字节
    data = client.recv(1024)
    # 打印接收数据
    option = client.recv(1)
    if option == b'1':
        op = input("请选择功能（1.信息传输 2.文件上传 3.退出)：")
        client.send(op.encode())
        if op == '1':
            msg_input = input("请输入发送的消息：")
            msg_input += '  /' + str(time.time()) + ' ' + str(client.getsockname())
            client.send(msg_input.encode())
        elif op == '2':
            file_name = input("请输入文件名称：")
            file_ = get_extension_after_last_dot(file_name)
            client.send(file_.encode())
            client.send(str(client.getsockname()).encode())
            with open(file_name, 'rb') as f:
                while True:
                    # 读取文件内容
                    data = f.read(1024)
                    if not data:
                        break
                    # 发送文件内容
                    client.send(data)
                    print(".", end='')
                    # 等待接收完成标志
                    response = client.recv(1024)
                    # 判断是否真正接收完成
                    if response != b'success':
                        break
                client.send('quit'.encode())
        elif op == '3':
            break
    elif option == b'2':
        sav = client.recv(1024)
        # 打印接收数据
        print(sav.decode())
        client.send('发送成功'.encode())
    else:
        continue