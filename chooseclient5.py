#基于模拟退火算法实现客户端分组组合优化选择

import numpy as np
import random
import math

# 定义客户端数量和其他参数
num_clients = 10
B = 3  # 类别数量
d = 10  # 数据集大小下限的基数
p_M = 0.25  # 剪枝率上限

# 随机生成客户端数据
clients = {}
for i in range(num_clients):
    q = random.randint(100, 200)  # 随机样本数量
    alpha = np.random.dirichlet(np.ones(B), size=1).flatten()  # 随机生成类别分布
    p = random.uniform(0.1, 0.7)  # 随机剪枝率
    clients[f'client{i+1}'] = {'q': q, 'alpha': alpha, 'p': p}
    print(clients[f'client{i+1}'])

def calculate_QCID(M):
    total_q = sum(clients[client]['q'] for client in M)
    sum_qqT = 0
    for n in M:
        for m in M:
            sum_qqT += clients[n]['q'] * clients[m]['q'] * np.dot(clients[n]['alpha'], clients[m]['alpha'])
    return sum_qqT / (total_q ** 2) - 1 / B

def get_neighbor(M):
    new_M = M.copy()
    if random.random() < 0.5 and len(new_M) > 1:
        new_M.remove(random.choice(list(new_M)))  # 随机移除一个客户端
    else:
        choices = list(set(clients.keys()) - new_M)
        if choices:
            new_M.add(random.choice(choices))  # 随机添加一个新的客户端
    return new_M
x
def simulated_annealing():
    current_M = set(random.sample(list(clients.keys()), random.randint(2, len(clients))))
    current_score = calculate_QCID(current_M)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        i = 1
        while i <= 100:
            new_M = get_neighbor(current_M)
            if sum(clients[client]['q'] for client in new_M) >= d**4 and all(clients[client]['p'] < p_M for client in new_M):
                new_score = calculate_QCID(new_M)
                ap = math.exp((current_score - new_score) / T)
                if new_score < current_score or random.random() < ap:
                    current_M = new_M
                    current_score = new_score
            i += 1
        T *= alpha

    return current_M, current_score

best_group, best_score = simulated_annealing()
print("Best group:", best_group)
print("Best QCID score:", best_score)
