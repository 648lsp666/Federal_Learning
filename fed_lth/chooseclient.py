import numpy as np
import random
from conf import conf
# client_info随机生成，测试用
def generate_client_info(num_clients, B, max_data_len, max_train_time, max_prune_ratio):
    client_info = {}
    for i in range(num_clients):
        data_dis = np.random.dirichlet(np.ones(B), size=1)[0].tolist()  # 生成一个随机的数据分布，和为1
        train_data_len = random.randint(1, max_data_len)  # 随机生成训练数据长度
        train_time = random.randint(1, max_train_time)  # 随机生成训练时间
        prune_ratio = round(random.uniform(0, max_prune_ratio), 2)  # 随机生成剪枝率，保留两位小数
        
        client_info[i] = {
            'id': i,
            'data_dis': data_dis,
            'train_data_len': train_data_len,
            'train_time': train_time,
            'prune_ratio': prune_ratio
        }
    return client_info

# 筛选符合条件的客户端
def filter_clients_by_prune_ratio(client_info, pM, remaining_clients):
    filtered_client_info = {k: v for k, v in client_info.items() if k in remaining_clients and v['prune_ratio'] <= pM}
    return filtered_client_info

# 计算QCID值的函数
def calculate_qcid(grouping, q_n, alpha_n, B):
    q_n_prime = np.array([q_n[i] for i in grouping])  # 当前分组中的q_n值
    alpha_n_prime = np.array([alpha_n[i] for i in grouping])  # 当前分组中的alpha_n值
    numerator = np.sum(np.outer(q_n_prime, q_n_prime) * np.dot(alpha_n_prime, alpha_n_prime.T))
    denominator = np.sum(q_n_prime) ** 2
    qcid = numerator / denominator - 1 / B
    return qcid

# 检查约束条件的函数
def check_constraints(grouping, prune_ratios, pM, group_size):
    prune_ratios_prime = np.array([prune_ratios[i] for i in grouping])
    return all(prune_ratios_prime < pM) and len(grouping) == group_size

# 模拟退火算法的主体
def simulated_annealing(client_info, remaining_clients, num_iterations, cooling_rate, pM, B, group_size):
    client_indices = list(client_info.keys())
    q_n = np.array([client_info[i]['train_data_len'] for i in client_indices])
    alpha_n = np.array([client_info[i]['data_dis'] for i in client_indices])
    prune_ratios = np.array([client_info[i]['prune_ratio'] for i in client_indices])
    
    current_T = 1.0
    best_grouping = random.sample(remaining_clients, min(group_size, len(remaining_clients)))
    while not check_constraints([client_indices.index(i) for i in best_grouping], prune_ratios, pM, group_size):
        best_grouping = random.sample(remaining_clients, min(group_size, len(remaining_clients)))
    
    best_qcid = calculate_qcid([client_indices.index(i) for i in best_grouping], q_n, alpha_n, B)
    
    for _ in range(num_iterations):
        operation = random.choice(['add', 'remove'])
        current_grouping = list(best_grouping)
        
        if operation == 'add' and len(current_grouping) < group_size:
            candidates = [c for c in remaining_clients if c not in current_grouping]
            if candidates:
                client_to_add = random.choice(candidates)
                current_grouping.append(client_to_add)
        elif operation == 'remove' and len(current_grouping) > group_size:
            client_to_remove = random.choice(current_grouping)
            current_grouping.remove(client_to_remove)
        
        if check_constraints([client_indices.index(i) for i in current_grouping], prune_ratios, pM, group_size):
            current_qcid = calculate_qcid([client_indices.index(i) for i in current_grouping], q_n, alpha_n, B)
            if current_qcid < best_qcid:
                best_grouping = current_grouping
                best_qcid = current_qcid
            else:
                delta_qcid = current_qcid - best_qcid
                if random.random() < np.exp(-delta_qcid / current_T):
                    best_grouping = current_grouping
        
        current_T *= cooling_rate
    
    return best_grouping, best_qcid

# 运行多次模拟退火算法，直到所有客户端都被分组
def multi_group_simulated_annealing(client_info, num_iterations, cooling_rate, initial_pM, delta_pM, B, group_size):
    all_groupings = []
    remaining_clients = list(client_info.keys())
    pM = initial_pM
    
    while remaining_clients:
        if len(remaining_clients) < group_size or pM>=100:  # 如果剩余客户数量小于group_size, 自成一组
            all_groupings.append(remaining_clients)
            break
        
        filtered_client_info = filter_clients_by_prune_ratio(client_info, pM, remaining_clients)
        if len(filtered_client_info)<group_size:  # 如果没有符合条件的客户端，增加 pM 值
            pM += delta_pM
            continue
        
        best_grouping, best_qcid = simulated_annealing(filtered_client_info, list(filtered_client_info.keys()), num_iterations, cooling_rate, pM, B, group_size)
        if not best_grouping:  # 如果没有找到有效的分组，增加 pM 值
            pM += delta_pM
            continue
        
        all_groupings.append(best_grouping)
        remaining_clients = [c for c in remaining_clients if c not in best_grouping]
        pM += delta_pM  # 增加 pM 值
    
    return all_groupings

def client_group(client_info):
    num_clients = len(client_info)
    group_size = int(num_clients * conf['k'])
    if group_size<=1:
        return [[id for id in client_info]]
    B = 4    
    initial_pM = 0.3
    delta_pM = 0.1
    num_iterations = 10000
    cooling_rate = 0.99
    groups = multi_group_simulated_annealing(client_info, num_iterations, cooling_rate, initial_pM, delta_pM, B, group_size)
    print(f'groups:{groups}')
    return groups
# 定义公共接口
__all__ = ['generate_client_info', 'filter_clients_by_prune_ratio', 'calculate_qcid', 'check_constraints', 'simulated_annealing', 'multi_group_simulated_annealing']

# 外部调用示例
if __name__ == "__main__":
    import pickle
    with open('result/client_info20.pkl','rb') as f:
        client_info=pickle.load(f)
    
    # 打印生成的客户端信息
    for client in client_info.values():
        print(client)

    all_groupings = client_group(client_info)
    print(f"All groupings: {all_groupings}")
