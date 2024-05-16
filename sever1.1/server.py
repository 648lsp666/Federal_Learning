# 非阻塞模块
import socketserver
import time
import torch
import torchvision.models as models

# 首先我们需要定义一个类
class my_socket_server(socketserver.BaseRequestHandler):
    # 首先执行setup方法，然后执行handle方法，最后执行finish方法
    # 如果handle方法报错，则会跳过
    # setup与finish无论如何都会执行
    # 一般只定义handle方法即可
    def setup(self):
        pass

    def handle(self):
        # 定义连接变量
        conn = self.request
        # 提示信息
        print("连接成功")
        # 发送消息定义
        msg = "已连接服务器!"
        # 发送消息
        conn.send(msg.encode())
        # 进入循环，不断接收客户端消息
        while True:
            # 接收客户端消息
            option = input("请选择功能（1.接收 2.发送)：")
            conn.send(option.encode())
            if option == '1':
                op = conn.recv(1024)
                if op == b'1':
                    data = conn.recv(1024)
                    # 打印消息
                    print(data.decode())
                    conn.send('接收成功'.encode())
                elif op == b'2':
                    flie_ = conn.recv(1024)
                    flie_ = flie_.decode()
                    whe = conn.recv(1024)
                    whe = whe.decode()
                    flie_name = str(time.time()) + whe + flie_
                    with open(flie_name, 'wb') as f:
                        f.write(b'')
                    while True:
                        data = conn.recv(1024)
                        if data == b'quit':
                            break
                        # 写入文件
                        with open(flie_name, 'ab') as f:
                            f.write(data)
                            # 接受完成标志
                            conn.send('success'.encode())
                    print("文件接收完成")
                    conn.send('文件接收成功'.encode())
                    if (flie_ == '.pt'):
                        model2 = torch.load(flie_name)
                        avg_state_dict = {}
                        for key in model1.state_dict():
                            avg_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2

                        # 创建新模型并加载平均参数
                        avg_model = models.resnet18()
                        avg_model.load_state_dict(avg_state_dict)

                        torch.save(avg_model, 'save.pt')
                        print("聚合成功")
                elif op == b'3':
                    break
                else:
                    conn.send('非法选项'.encode())
            elif option == '2':
                conn.send('这是服务信息'.encode())
                data = conn.recv(1024)
                # 打印消息
                print(data.decode())
                conn.send('接收成功'.encode())
        conn.close()

    def finish(self):
        pass


if __name__ == "__main__":
    model1 = torch.load('save.pt')
    # 提示信息
    print("初始化完毕，正在等待接收数据。。。。。。")
    # 创建多线程实例
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 8888), my_socket_server)
    # 开启多线程，等待连接
    server.serve_forever()