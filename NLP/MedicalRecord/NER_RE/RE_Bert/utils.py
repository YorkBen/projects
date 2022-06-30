import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score


def load_data():
    """
    加载训练数据
    """
    lines = []
    with open('data.txt', "r") as f:
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


def text_tokenize(model_name, sentences, max_length):
    """
    文本转化为向量
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    sentences_tokened = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids, attention_mask = sentences_tokened['input_ids'], sentences_tokened['attention_mask']
    print("sentences tokenize finished...")
    return input_ids, attention_mask


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

class TextLabelToDataset(Dataset):
    """
    使用文本和特征数据作为标签
    """
    def __init__(self, input_ids, attention_mask, pos_pair, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pos_pair = torch.IntTensor(pos_pair)
        self.labels = torch.Tensor(labels).to(torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.pos_pair[index], self.labels[index]
