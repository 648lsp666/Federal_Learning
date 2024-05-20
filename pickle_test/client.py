import socket
import pickle
import struct
import torchvision

def send_data(host, port, data):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # 序列化数据
    data_bytes = pickle.dumps(data)

    # 发送数据大小
    data_size = len(data_bytes)
    client_socket.sendall(struct.pack('>I', data_size))

    # 发送数据内容
    client_socket.sendall(data_bytes)

    client_socket.close()

if __name__ == "__main__":
    data = '123'
    send_data('localhost', 12345, data)
