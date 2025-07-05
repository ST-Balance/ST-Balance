# 1
# SD
# 4.242380304914114
# 5.483906782899025
# 7
# SD
# 5.436077359137321
# 5.485268447447922
import numpy as np
from scipy.stats import iqr, entropy
import argparse
import ast

def auto_bin_edges(data, method="fd"):
    """
    自动计算分箱边界
    :param data: 输入数据（一维数组）
    :param method: 分箱规则 ("sturges", "fd", or "sqrt")
    :return: 分箱边界数组
    """
    N = len(data)
    if method == "sturges":
        num_bins = int(np.ceil(np.log2(N) + 1))
    elif method == "fd":
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * N ** (-1 / 3)
        if bin_width <= 0 or np.isinf(bin_width):  # 防止 bin_width 为 0 或无效
            num_bins = int(np.ceil(np.sqrt(N)))  # 退回到默认分箱规则
        else:
            num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    elif method == "sqrt":
        num_bins = int(np.ceil(np.sqrt(N)))
    else:
        raise ValueError("Unsupported binning method")
    num_bins = max(10, num_bins)  # 分箱数至少为 1
    return np.linspace(data.min(), data.max(), num_bins + 1)


def calculate_bins(data, method="fd"):
    """
    根据分箱规则计算分箱数量
    :param data: 输入数据（一维数组）
    :param method: 分箱规则 ("sturges", "fd", "sqrt")
    :return: 分箱数量
    """
    edges = auto_bin_edges(data, method=method)
    return len(edges) - 1


def calculate_time_entropy(data, bin_method="fd"):
    """
    计算时间熵：对每个传感器的时间序列计算熵，然后取平均值
    :param data: 交通数据矩阵 (N x M)，N 是时间步，M 是传感器数量
    :param bin_method: 分箱规则 ("sturges", "fd", "sqrt")
    :return: 时间熵
    """
    time_entropies = []
    for sensor_data in data.T:  # 遍历每个传感器的时间序列
        # 计算概率分布
        unique, counts = np.unique(sensor_data, return_counts=True)
        probabilities = counts / counts.sum()
        # 计算熵
        time_entropies.append(entropy(probabilities))
    return np.mean(time_entropies)


def calculate_space_entropy(data, bin_method="fd"):
    """
    计算空间熵：对每个时间步的空间分布计算熵，然后取平均值
    :param data: 交通数据矩阵 (N x M)，N 是时间步，M 是传感器数量
    :param bin_method: 分箱规则 ("sturges", "fd", "sqrt")
    :return: 空间熵
    """
    space_entropies = []
    for time_data in data:  # 遍历每个时间步的空间分布
        # 计算概率分布
        unique, counts = np.unique(time_data, return_counts=True)
        probabilities = counts / counts.sum()
        # 计算熵
        space_entropies.append(entropy(probabilities))
    return np.mean(space_entropies)


def calculate_joint_entropy(data, time_bin_method="fd", space_bin_method="fd"):
    """
    计算时空联合熵
    :param data: 交通数据矩阵 (N x M)，N 是时间步，M 是传感器数量
    :param time_bin_method: 时间维度分箱规则 ("sturges", "fd", "sqrt")
    :param space_bin_method: 空间维度分箱规则 ("sturges", "fd", "sqrt")
    :return: 时空联合熵
    """
    N, M = data.shape

    # 对时间维度和空间维度分别计算分箱边界
    time_edges = auto_bin_edges(data.flatten(), method=time_bin_method)
    space_edges = auto_bin_edges(data.flatten(), method=space_bin_method)

    # 初始化联合频率矩阵
    time_bins = len(time_edges) - 1
    space_bins = len(space_edges) - 1
    joint_freq = np.zeros((time_bins, space_bins))

    # 统计联合分布
    for i in range(N):
        for j in range(M):
            time_bin = np.digitize(data[i, j], time_edges) - 1
            space_bin = np.digitize(data[i, j], space_edges) - 1
            # 确保分箱索引合法
            time_bin = min(time_bin, time_bins - 1)
            space_bin = min(space_bin, space_bins - 1)
            joint_freq[time_bin, space_bin] += 1

    # 归一化为联合概率分布
    joint_prob = joint_freq / np.sum(joint_freq)

    # 计算联合熵
    joint_entropy = -np.nansum(joint_prob * np.log(joint_prob + 1e-9))  # 避免 log(0)
    return joint_entropy


# 滑动窗口熵计算
def sliding_window_entropy(data, window_size, step_size):
    time_entropies = []
    space_entropies = []
    joint_entropies = []
    N, M = data.shape
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        window_data = data[start_idx:start_idx + window_size]
        # 时间熵计算（针对每个传感器的时间序列）
        sensor_entropies = []
        for sensor_idx in range(M):  # 遍历每个传感器
            sensor_data = window_data[:, sensor_idx]
            unique, counts = np.unique(sensor_data, return_counts=True)
            probabilities = counts / counts.sum()
            sensor_entropies.append(entropy(probabilities))
        time_entropies.append(np.mean(sensor_entropies))  # 平均时间熵

        # 空间熵计算（针对每个时间步的传感器值）
        timestep_entropies = []
        for time_idx in range(window_size):  # 遍历每个时间步
            time_data = window_data[time_idx, :]
            unique, counts = np.unique(time_data, return_counts=True)
            probabilities = counts / counts.sum()
            timestep_entropies.append(entropy(probabilities))
        space_entropies.append(np.mean(timestep_entropies))  # 平均空间熵

    return np.array(time_entropies), np.array(space_entropies), np.array(joint_entropies)


def main():
    parser = argparse.ArgumentParser(description="Calculate entropies for datasets.")
    parser.add_argument("--dataset", type=str, help="List of dataset names")
    parser.add_argument("--horizon", type=str, help="List of horizon values")
    args = parser.parse_args()

    datasets = ast.literal_eval(args.dataset)
    horizon = ast.literal_eval(args.horizon)


    if not isinstance(datasets, list) or not all(isinstance(item, str) for item in datasets):
        print("Error: --dataset must be a list of strings.")
        return

    if not isinstance(horizon, list) or not all((isinstance(item, float) or isinstance(item, int)) for item in horizon) :
        print("Error: --horizon must be a list of integers.")
        return

    for i in horizon:
        for dataset in datasets:
            try:
                data = np.load(f'data/{dataset}/{dataset}.npz')['data'][:, :, 0]
            except FileNotFoundError:
                print(f"Error: Data file for dataset {dataset} not found.")
                continue

            if 'PEMS' in dataset:
                data = (data - np.mean(data)) / np.std(data)

            T = 5
            if 'PEMS' in dataset:
                T = 5  # 每个窗口包含10个时间步
            else:
                T = 15
            days = i
            window_size = int(days * ((24 * 60) / T))
            step_size = 12  # 每次滑动1个时间步

            print(i)
            print(dataset)
            # 计算滑动窗口时间熵、空间熵和联合熵
            time_entropies, space_entropies, joint_entropies = sliding_window_entropy(data, window_size, step_size)
            print(np.mean(time_entropies))
            print(np.mean(space_entropies))

if __name__ == "__main__":
    main()