import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from transformers import BertModel
import time
from sys import argv
from utils import *

import random
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import joblib
import pickle

torch.cuda.empty_cache()

# 参数
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext-large", 1024, r"models\torch\large", 1
model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext", 768, r"models\torch\common", 128
# model_name, hidden_size, model_output, BATCH_SIZE = "bert-base-chinese", 768, r"models\torch\google", 8

# model_name = "hfl/chinese-roberta-wwm-ext-large"
# input_file, num_cls = r'data\data_no0.csv', 9      # 测试数据
# input_file, num_cls = r'data\data_no_9.csv', 9      # 急性 + 慢性
# input_file, num_cls = r'data\data_no_8.csv', 8      # 急性 + 慢性 去掉GA18
# input_file, num_cls = r'data\data_no_6.csv', 6    # 急性阑尾炎
max_length = 500
epochs = 100
feature_size =  4
hidden_size1 = 1024
hidden_size2 = 2048
hidden_size3 = 512
name = "mix"
# id_map, num_cls = id_map_6, 6
# idc11_id, num_cls = "DB10", 2

multilbl_dict = {}
with open('data/multilbl_ind.txt') as f:
    for idx, line in enumerate(f.readlines()):
        multilbl_dict[line.replace('\n', '')] = idx

num_cls = len(multilbl_dict)


def load_data_file(file_path, n_labels, separator='	', skip_head=True):
    """
    加载数据文件，文件格式：
    f1, f2, f3, ... fn, lbl1, lbl2, lbl3...
    """
    X, y, feature_names, label_names = [], [], [], []
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            arr = line.strip().split(separator)
            if idx == 0:
                feature_names = arr[:-n_labels]
                label_names = arr[-n_labels:-1]
            else:
                lines.append(arr)

#     random.shuffle(lines)

    X = np.array([[float(e) for e in arr[:-n_labels]] for arr in lines])
    #labels = [[int(1) if float(e) > 1 else float(e) for e in arr[-n_labels:]] for arr in lines]
    y = np.array([multilbl_dict[' '.join(arr)] for arr in lines], dtype=np.int64)

    return X, y, feature_names, label_names


if __name__ == "__main__":
    n_labels = 11
    X_train, y_train, _, _ = load_data_file(r'data/训练_全特征_多诊断.txt', n_labels)
    n_features = len(X_train[0])
    X_test, y_test, _, _ = load_data_file(r'data/测试_全特征_多诊断.txt', n_labels)
