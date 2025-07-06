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
    N = len(data)
    if method == "sturges":
        num_bins = int(np.ceil(np.log2(N) + 1))
    elif method == "fd":
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * N ** (-1 / 3)
        if bin_width <= 0 or np.isinf(bin_width):  
            num_bins = int(np.ceil(np.sqrt(N)))  
        else:
            num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    elif method == "sqrt":
        num_bins = int(np.ceil(np.sqrt(N)))
    else:
        raise ValueError("Unsupported binning method")
    num_bins = max(10, num_bins)  
    return np.linspace(data.min(), data.max(), num_bins + 1)


def calculate_bins(data, method="fd"):
    edges = auto_bin_edges(data, method=method)
    return len(edges) - 1


def calculate_time_entropy(data, bin_method="fd"):
    time_entropies = []
    for sensor_data in data.T:
        unique, counts = np.unique(sensor_data, return_counts=True)
        probabilities = counts / counts.sum()
        time_entropies.append(entropy(probabilities))
    return np.mean(time_entropies)


def calculate_space_entropy(data, bin_method="fd"):
    space_entropies = []
    for time_data in data:  
        unique, counts = np.unique(time_data, return_counts=True)
        probabilities = counts / counts.sum()
        space_entropies.append(entropy(probabilities))
    return np.mean(space_entropies)


def calculate_joint_entropy(data, time_bin_method="fd", space_bin_method="fd"):
    N, M = data.shape

    time_edges = auto_bin_edges(data.flatten(), method=time_bin_method)
    space_edges = auto_bin_edges(data.flatten(), method=space_bin_method)

    time_bins = len(time_edges) - 1
    space_bins = len(space_edges) - 1
    joint_freq = np.zeros((time_bins, space_bins))

    for i in range(N):
        for j in range(M):
            time_bin = np.digitize(data[i, j], time_edges) - 1
            space_bin = np.digitize(data[i, j], space_edges) - 1
            time_bin = min(time_bin, time_bins - 1)
            space_bin = min(space_bin, space_bins - 1)
            joint_freq[time_bin, space_bin] += 1

    joint_prob = joint_freq / np.sum(joint_freq)

    joint_entropy = -np.nansum(joint_prob * np.log(joint_prob + 1e-9))
    return joint_entropy


def sliding_window_entropy(data, window_size, step_size):
    time_entropies = []
    space_entropies = []
    joint_entropies = []
    N, M = data.shape
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        window_data = data[start_idx:start_idx + window_size]
        sensor_entropies = []
        for sensor_idx in range(M): 
            sensor_data = window_data[:, sensor_idx]
            unique, counts = np.unique(sensor_data, return_counts=True)
            probabilities = counts / counts.sum()
            sensor_entropies.append(entropy(probabilities))
        time_entropies.append(np.mean(sensor_entropies))  

        timestep_entropies = []
        for time_idx in range(window_size): 
            time_data = window_data[time_idx, :]
            unique, counts = np.unique(time_data, return_counts=True)
            probabilities = counts / counts.sum()
            timestep_entropies.append(entropy(probabilities))
        space_entropies.append(np.mean(timestep_entropies))  

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
                T = 5  
            else:
                T = 15
            days = i
            window_size = int(days * ((24 * 60) / T))
            step_size = 12 

            print(i)
            print(dataset)
            time_entropies, space_entropies, joint_entropies = sliding_window_entropy(data, window_size, step_size)
            print(np.mean(time_entropies))
            print(np.mean(space_entropies))

if __name__ == "__main__":
    main()
