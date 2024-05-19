import socket
import pickle
import struct
import torchvision

def receive_data(conn):
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
        print('.',end='')

    return pickle.loads(data)

def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server started and listening on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Connection from {addr}")

    # 接收数据
    data_dict = receive_data(conn)
    if data_dict:
        print(f"Received data: {data_dict}")

    conn.close()

if __name__ == "__main__":
    start_server('localhost', 12345)
