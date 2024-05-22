import numpy as np
import random

# client_info随机生成，测试用
def generate_client_info(num_clients, B, max_data_len, max_train_time, max_prune_ratio):
    client_info = []
    for i in range(num_clients):
        data_dis = np.random.dirichlet(np.ones(B), size=1)[0].tolist()  # 生成一个随机的数据分布，和为1
        train_data_len = random.randint(1, max_data_len)  # 随机生成训练数据长度
        train_time = random.randint(1, max_train_time)  # 随机生成训练时间
        prune_ratio = round(random.uniform(0, max_prune_ratio), 2)  # 随机生成剪枝率，保留两位小数
        
        client_info.append({
            'id': i,
            'data_dis': data_dis,
            'train_data_len': train_data_len,
            'train_time': train_time,
            'prune_ratio': prune_ratio
        })
    return client_info

# 计算QCID值的函数
def calculate_qcid(grouping, q_n, alpha_n, B):
    q_n_prime = np.array([q_n[i] for i in grouping])  # 当前分组中的q_n值
    alpha_n_prime = np.array([alpha_n[i] for i in grouping])  # 当前分组中的alpha_n值
    numerator = np.sum(np.outer(q_n_prime, q_n_prime) * np.dot(alpha_n_prime, alpha_n_prime.T))
    denominator = np.sum(q_n_prime) ** 2
    qcid = numerator / denominator - 1 / B
    return qcid

# 检查约束条件的函数
def check_constraints(grouping, q_n, pM, d, prune_ratios):
    q_n_prime = np.array([q_n[i] for i in grouping])
    prune_ratios_prime = np.array([prune_ratios[i] for i in grouping])
    data_size = np.sum(q_n_prime)
    return all(prune_ratios_prime < pM) and data_size >= d**4

# 模拟退火算法的主体
def simulated_annealing(client_info, num_iterations, cooling_rate, pM, d, B):
    num_clients = len(client_info)
    q_n = np.array([client['train_data_len'] for client in client_info])
    alpha_n = np.array([client['data_dis'] for client in client_info])
    prune_ratios = np.array([client['prune_ratio'] for client in client_info])
    
    current_T = 1.0
    # 随机生成初始分组,只在剪枝率满足小于PM的条件下生成
    best_grouping = random.sample([i for i in range(num_clients) if prune_ratios[i] <= pM], random.randint(1, num_clients))
    best_qcid = calculate_qcid(best_grouping, q_n, alpha_n, B)
    best_prune_ratios = prune_ratios[best_grouping]
    
    for _ in range(num_iterations):
        # 随机选择一个操作：增加、减少或保持不变
        operation = random.choice(['add', 'remove'])
        current_grouping = list(best_grouping)  # 从当前最佳分组开始
        
        if operation == 'add' and len(current_grouping) < num_clients:
            # 添加一个客户端,只添加满足剪枝率满足小于PM的客户端
            candidates = [c for c in range(num_clients) if c not in current_grouping and prune_ratios[c] <= pM]
            if candidates:
                client_to_add = random.choice(candidates)
                current_grouping.append(client_to_add)
        elif operation == 'remove' and current_grouping:
            # 移除一个客户端
            client_to_remove = random.choice(current_grouping)
            current_grouping.remove(client_to_remove)
        
        # 检查新的分组是否满足约束条件
        if check_constraints(current_grouping, q_n, pM, d, prune_ratios):
            current_qcid = calculate_qcid(current_grouping, q_n, alpha_n, B)
            if current_qcid < best_qcid:
                best_grouping = current_grouping
                best_qcid = current_qcid
                best_prune_ratios = prune_ratios[best_grouping]
            else:
                delta_qcid = current_qcid - best_qcid
                if random.random() < np.exp(-delta_qcid / current_T):
                    best_grouping = current_grouping
                    best_prune_ratios = prune_ratios[best_grouping]
        
        current_T *= cooling_rate  # 降温
    
    return best_grouping, best_qcid, best_prune_ratios

# 定义公共接口
__all__ = ['generate_client_info', 'calculate_qcid', 'check_constraints', 'simulated_annealing']

# 外部调用示例
if __name__ == "__main__":
    # 外部调用时传入真实的客户端信息，测试用
    num_clients = 20  # 客户端数量
    B = 4  # 类别标签的数量
    max_data_len = 150  # 训练数据长度的最大值
    max_train_time = 200  # 训练时间的最大值
    max_prune_ratio = 0.9  # 剪枝率的最大值
    
    pM = 0.5  # 客户端组M的剪枝率上限
    d = 10  # 客户端组M的数据集样本数量下限
    num_iterations = 10000 # 算法迭代次数
    cooling_rate = 0.99 # 降温速度

     # 生成随机的客户端信息
    client_info = generate_client_info(num_clients, B, max_data_len, max_train_time, max_prune_ratio)
    
    # 打印生成的客户端信息
    for client in client_info:
        print(client)

    # 运行模拟退火算法
    best_grouping, min_qcid, best_prune_ratios = simulated_annealing(client_info, num_iterations=10000, cooling_rate=0.99, pM=3, d=10, B=4)
    print(f"Best grouping: {best_grouping}")
    print(f"Minimum QCID: {min_qcid}")
    print(f"Prune ratios of best grouping: {best_prune_ratios}")
