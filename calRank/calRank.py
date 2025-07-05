import numpy as np
import pickle
import os
import argparse
import ast

# 从pkl文件中读取矩阵
def load_matrix_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        matrix = pickle.load(f)
    return matrix

# 从npy文件中读取矩阵
def load_matrix_from_npy(file_path):
    matrix = np.load(file_path)
    return matrix

# 计算矩阵的秩
def calculate_matrix_rank(matrix):
    rank = np.linalg.matrix_rank(matrix)
    return rank

# 主函数
def main(dataset_names):
    for dataset_name in dataset_names:
        # 根据数据集名称决定读取pkl文件还是npy文件
        if "PEMS" in dataset_name:
            file_path = f'data/{dataset_name}/adj_mx.pkl'  # pkl文件路径
            load_matrix = load_matrix_from_pkl
        else:
            file_path = f'data/{dataset_name}/adj_{dataset_name}.npy'  # npy文件路径
            load_matrix = load_matrix_from_npy

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，请检查路径。")
            continue

        # 从文件中读取矩阵
        matrix = load_matrix(file_path)
        print(f"\n数据集 {dataset_name} 的矩阵为：")
        print(matrix)

        # 计算矩阵的秩
        rank = calculate_matrix_rank(matrix)
        print(f"数据集 {dataset_name} 的矩阵的秩为：", rank)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算多个数据集的矩阵秩')
    parser.add_argument('--dataset', type=str, help='数据集名称列表，格式为 \"[\'PEMS08\', \'SD\']\"')
    args = parser.parse_args()

    # 将字符串转换为列表
    dataset_names = ast.literal_eval(args.dataset)

    # 检查输入是否为列表
    if not isinstance(dataset_names, list):
        print("数据集名称必须是一个列表，例如 ['PEMS08', 'SD']")
        exit(1)

    main(dataset_names)