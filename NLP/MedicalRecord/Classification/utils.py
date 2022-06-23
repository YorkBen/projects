import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score


id_map_6 = ["GB70", "DC31", "DB10", "DA91", "JA01", "DC11"]
id_map_8 = ["GB70", "DC31", "DB10", "DA91", "JA01", "DC11", "DA42", "2E86"]
id_map_9 = ["GB70", "DC31", "DB10", "DA91", "JA01", "DC11", "DA42", "2E86", "GA18"]

def load_data(id_map, input_file=r'data/medical_record_merge_labeled.txt', has_title=True):
    """
    加载训练数据
    """
    lines = []
    with open(input_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if has_title and idx == 0:
                continue
            lines.append(line)

    random.shuffle(lines)
    print('data total lines: %s' % len(lines))

    # create id set
    id_set = set(id_map)
    row_len = len(lines[0].split(','))
    sentences, features, labels = [], [], []
    for line in lines:
        arr = line.split(',')
        if len(arr) != row_len:
            print('illegal line: ' + line)
            exit()

        if arr[2] in id_set:
            sentences.append(arr[1])
            features.append([int(e) for e in arr[4:]])
            labels.append(id_map.index(arr[2]))

    return sentences, features, labels, len(features[0])


def load_data_one_id(id, input_file=r'data/medical_record_merge_labeled.txt', has_title=True, pn_balance=True):
    """
    一个作为正例，其它作为反例
    """
    lines = []
    with open(input_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if has_title and idx == 0:
                continue
            arr = line.split(',')
            if len(arr) != 12:
                print('illegal line: ' + line)
                continue
            lines.append(line)

    random.shuffle(lines)
    print('data total lines: %s' % len(lines))

    # create id set
    p_count, n_count = 0, 0
    lines_p, lines_n = [], []
    for line in lines:
        arr = line.split(',')
        if arr[10] == id:
            lines_p.append(line)
            p_count = p_count + 1
        else:
            lines_n.append(line)
            n_count = n_count + 1


    print('positive vs negtive sample num: %s:%s' % (p_count, n_count))
    if pn_balance:
        print('balance positive and negtive samples...')
        lines = []
        if p_count >= n_count:
            lines = lines_n
            lines.extend(lines_p[:n_count])
        else:
            lines = lines_p
            lines.extend(lines_n[:p_count])
        random.shuffle(lines)
        print('positive vs negtive sample num: %s:%s' % (len(lines) // 2, len(lines) // 2))

    sentences, features, labels = [], [], []
    for line in lines:
        arr = line.split(',')
        sentences.append(arr[1])
        features.append([int(e) for e in arr[2:10]])
        if arr[10] == id:
            labels.append(1)
        else:
            labels.append(0)

    return sentences, features, labels


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
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


class TextDataToDataset(Dataset):
    """
    只使用文本和标签数据
    """
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

class FeatureDataToDataset(Dataset):
    """
    只是用特征和标签数据
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.features[index], self.labels[index]

class MixDataToDataset(Dataset):
    """
    使用文本和特征以及标签数据
    """
    def __init__(self, input_ids, attention_mask, features, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.features[index], self.labels[index]

class TextLabelToDataset(Dataset):
    """
    使用文本和特征数据作为标签
    """
    def __init__(self, input_ids, attention_mask, features, idx):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.features = features
        self.labels = torch.tensor(np.array(features)[:, idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]
