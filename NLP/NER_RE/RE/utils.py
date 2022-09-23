import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    加载训练数据
    """
    lines = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
            lines.append(line)

    random.shuffle(lines)
    print('data total lines: %s' % len(lines))

    # create id set
    row_len = len(lines[0].split('	'))
    sentences, pos_pairs, labels = [], [], []
    for line in lines:
        arr = line.split('	')
        if len(arr) != row_len:
            print('illegal line: ' + line)
            exit()

        sentences.append(arr[0])
        pos_pairs.append([[int(arr[1]), int(arr[2])], [int(arr[3]), int(arr[4])]])
        labels.append(int(arr[5]))

    return sentences, pos_pairs, labels


def choose_device(model):
    #获取gpu和cpu的设备信息
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Use Device: cuda")
        if torch.cuda.device_count() > 1:
            print("Use multiply gpus: %s" % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        print("Use Device: cpu")

    return device

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)
