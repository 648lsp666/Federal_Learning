import numpy as np
import random
import math

# 定义客户端数量和其他参数
num_clients = 10
B = 3  # 类别数量
d = 3  # 数据集大小下限的基数
p_M = 0.25  # 剪枝率上限

# 随机生成客户端数据
clients = {}
for i in range(num_clients):
    q = random.randint(100, 200)  # 随机样本数量
    alpha = np.random.dirichlet(np.ones(B), size=1).flatten()  # 随机生成类别分布
    p = random.uniform(0.1, 0.25)  # 随机剪枝率
    clients[f'client{i+1}'] = {'q': q, 'alpha': alpha, 'p': p}  # 存储客户端数据

# 打印每个客户端的信息
print("Clients' information:")
for client, info in clients.items():
    print(f"{client}: samples={info['q']}, alpha={info['alpha']}, pruning rate={info['p']}")

# 计算给定客户端组 M 的 QClD 值
def calculate_QCID(M):
    total_q = sum(clients[client]['q'] for client in M)  # 总样本数量
    sum_qqT = 0
    for n in M:
        for m in M:
            sum_qqT += clients[n]['q'] * clients[m]['q'] * np.dot(clients[n]['alpha'], clients[m]['alpha'])
    return sum_qqT / (total_q ** 2) - 1 / B

# 生成一个邻居
def get_neighbor(M):
    new_M = M.copy()
    if random.random() < 0.5 and len(new_M) > 1:
        # 随机移除一个客户端
        client_to_remove = random.choice(list(new_M))
        new_M.remove(client_to_remove)
    else:
        # 随机添加一个新的客户端
        choices = list(set(clients.keys()) - new_M)
        if choices:
            client_to_add = random.choice(choices)
            new_M.add(client_to_add)
    return new_M

# 检查客户端组 M 是否满足限制条件
def valid_group(M):
    total_q = sum(clients[client]['q'] for client in M)
    # 样本数量下限和剪枝率上限检查
    return total_q >= d**4 and all(clients[client]['p'] < p_M for client in M)

# 初始化一个满足限制条件的客户端组
def init_valid_group(max_attempts=1000000):
    attempts = 0
    while attempts < max_attempts:
        initial_group = set(random.sample(list(clients.keys()), random.randint(2, len(clients))))
        if valid_group(initial_group):
            return initial_group
        attempts += 1
    raise Exception("Failed to initialize a valid group after many attempts")

# 模拟退火算法实现客户端组选择
def simulated_annealing():

    current_M = init_valid_group()  # 初始化一个有效的客户端组
    current_score = calculate_QCID(current_M)  # 计算初始组的 QClD 值
    T = 1.0  # 初始温度
    T_min = 0.00001  # 最低温度
    alpha = 0.9  # 温度下降率

    while T > T_min:
        i = 1
        while i <= 100:
            new_M = get_neighbor(current_M)  # 生成邻居
            if valid_group(new_M):
                new_score = calculate_QCID(new_M)  # 计算新邻居组的 QClD 值
                ap = math.exp((current_score - new_score) / T)  # 计算接受概率
                if new_score < current_score or random.random() < ap:
                    current_M = new_M  # 接受新邻居组
                    current_score = new_score  # 更新当前最优值
                    # print(f"Accepted new group: {current_M} with score: {current_score}")
            i += 1
        T *= alpha  # 降低温度
        # print(f"Temperature decreased to {T}")

    return current_M, current_score

best_group, best_score = simulated_annealing()  # 执行模拟退火算法
print("Best group:", best_group)
print("Best QCID score:", best_score)
