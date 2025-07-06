import numpy as np
import pickle
import os
import argparse
import ast

def load_matrix_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        matrix = pickle.load(f)
    return matrix

def load_matrix_from_npy(file_path):
    matrix = np.load(file_path)
    return matrix

def calculate_matrix_rank(matrix):
    rank = np.linalg.matrix_rank(matrix)
    return rank

def main(dataset_names):
    for dataset_name in dataset_names:
        if "PEMS" in dataset_name:
            file_path = f'data/{dataset_name}/adj_mx.pkl' 
            load_matrix = load_matrix_from_pkl
        else:
            file_path = f'data/{dataset_name}/adj_{dataset_name}.npy' 
            load_matrix = load_matrix_from_npy


        if not os.path.exists(file_path):
            print(f"file {file_path} not exist。")
            continue

        matrix = load_matrix(file_path)
        print(f"\ndataset {dataset_name} is：")
        print(matrix)

        rank = calculate_matrix_rank(matrix)
        print(f"dataset {dataset_name} rank is：", rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate rank')
    parser.add_argument('--dataset', type=str, help='Dataset name list, format: \"[\'PEMS08\', \'SD\']\"')
    args = parser.parse_args()

    dataset_names = ast.literal_eval(args.dataset)

    if not isinstance(dataset_names, list):
        print("The dataset name must be a list，例如 ['PEMS08', 'SD']")
        exit(1)

    main(dataset_names)
