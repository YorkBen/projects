import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
import random
from sys import argv
from utils import load_data as load_txt_data
from utils import id_map_6

torch.cuda.empty_cache()

# 参数
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext-large", 1024, r"output/models\torch\large", 1
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext", 768, r"output/models/clss/common", 8
model_name, hidden_size, BATCH_SIZE = "BertModels/medical-roberta-wwm", 768, 8
# model_name, hidden_size, model_output, BATCH_SIZE = "BertModels/test-mlm", 768, r"output/models/clss/common", 8

# input_file, num_cls = r'data/processed2.txt', 2    # 急性阑尾炎
# input_file, num_cls = r'data/train_data_20220424.txt', 3  # 恶心呕吐
input_file, num_cls = r'data/放射痛.txt', 3

max_length = 500
epochs = 100
model_dir = 'text'
result_file = r'data/featureClassifyResult2.txt'


class DataToDataset(Dataset):
    def __init__(self, tokenizer, text_a_arr, text_b_arr, labels, max_length):
        if text_b_arr is not None:
            sentences_tokened = tokenizer(text_a_arr, text_b_arr, padding='max_length', truncation='only_first', max_length=max_length, return_tensors='pt')
            # sentences_tokened = tokenizer(sentences, cmps, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        else:
            sentences_tokened = tokenizer(text_a_arr, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

        self.input_ids = sentences_tokened['input_ids']
        self.attention_mask = sentences_tokened['attention_mask']
        self.token_type_ids = sentences_tokened['token_type_ids']
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.labels[index]


class BertTextClassficationModel(torch.nn.Module):
    def __init__(self):
        super(BertTextClassficationModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dense = torch.nn.Linear(hidden_size, num_cls)
        self.dropout = torch.nn.Dropout(0.5)
        # self.frozen_bert = False

    def forward(self, ids, mask, types):
        # if self.frozen_bert:
        #     with torch.no_grad():
        #         outputs = self.bert(input_ids=ids, attention_mask=mask)
        # else:
        outputs = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=types)

        logits = self.dense(self.dropout(outputs.pooler_output))
        return logits

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def writeToFile(file_path, line):
    """
    将内容追加到文件
    """
    with open(file_path, "a+") as f:
        f.write(line + '\n')

def train(datasets, device):
    print('training model...')
    train_size = int(len(datasets) * 0.8)
    val_size = len(datasets) - train_size
    print("[train size, test size]: ", [train_size, val_size])
    train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # 模型
    mymodel = BertTextClassficationModel()
    mymodel.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.0001, weight_decay=1e-2)

    # 训练
    best_acc1 = 0
    best_modelname = ''
    print_num = 0
    no_improve_num = 0
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0.0
        train_acc = 0.0
        mymodel.train()
        for i, data in enumerate(train_loader):
            input_ids, attention_mask, types, labels = [elem.to(device) for elem in data]
            # # 调试
            # if print_num == 0:
            #     print(torch.cuda.memory_stats())
            #     print_num = 1

            #优化器置零
            optimizer.zero_grad()

            #得到模型的结果
            out = mymodel(input_ids, attention_mask, types)
            #计算误差
            loss = loss_func(out, labels)
            train_loss += loss.item()
            #误差反向传播
            loss.backward()
            #更新模型参数
            optimizer.step()
            #计算acc
            out = out.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_acc += flat_accuracy(out,labels)

        end = time.time()
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        print("Epoch: %d/%d, Training Loss:%f, Acc:%f, elapsed: %f" %(epoch, epochs, train_loss, train_acc, end-start))


        val_loss = 0
        val_acc = 0
        mymodel.eval()
        start = time.time()
        for j, batch in enumerate(val_loader):
            val_input_ids, val_attention_mask, val_types, val_labels = [elem.to(device) for elem in batch]
            with torch.no_grad():
                pred = mymodel(val_input_ids, val_attention_mask, val_types)
                val_loss += loss_func(pred, val_labels)
                pred = pred.detach().cpu().numpy()
                val_labels = val_labels.detach().cpu().numpy()
                val_acc += flat_accuracy(pred,val_labels)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        end = time.time()
        print("\t      Evalaton loss:%f, Acc:%f, elapsed: %f" %(val_loss, val_acc, end-start))

        if val_acc > best_acc1:
            no_improve_num = 0
            best_acc1 = val_acc
            best_modelname = r'output/models/%s/%s_%s.pth' % (model_dir, num_cls, int(best_acc1 * 10000))
            torch.save(mymodel, best_modelname)
        else:
            no_improve_num = no_improve_num + 1
            if no_improve_num > 3:
                print("no improve more than: %s, exit training. best accuracy: %f" % (no_improve_num, best_acc1))
                writeToFile(result_file, "Best accuracy: %f" % (best_acc1))
                return

def load_data():
    """
    加载恶心呕吐的模型
    """
    lines = []
    with open(input_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx > 0:
                lines.append(line)
    random.shuffle(lines)
    print('data rows: ', len(lines))
    max_sent_len, mean_sent_len = 0, 0
    sentences, labels = [], []
    row_field_num = len(lines[0].split(','))
    label_field_idx = 16
    n_0, n_1, n_2 = 0, 0, 0
    max_pos, mean_pos = 0, 0
    for line in lines:
        arr = line.strip().split(',')
        if len(arr) != row_field_num:
            print('illegal line: ' + line)
        else:
            sent_len = len(arr[1])
            if sent_len > max_sent_len:
                max_sent_len = sent_len
            mean_sent_len = mean_sent_len + sent_len

            if len(arr[1]) > 490:
                pos1 = max(arr[1].find('恶心'), arr[1].find('呕吐'))
                pos2 = max(arr[1].rfind('恶心'), arr[1].rfind('呕吐'))
                if pos2 - pos1 < 490:
                    sent = arr[1][max(pos1 - (pos2 - pos1) // 2, 0):490]
                else:
                    sent = arr[1][pos1:pos1+490]
                print(sent)
                sentences.append(sent)
            else:
                sentences.append(arr[1])

            labels.append(int(arr[label_field_idx]))
            if arr[label_field_idx] == "0":
                n_0 = n_0 + 1
            elif arr[label_field_idx] == "1":
                n_1 = n_1 + 1
            elif arr[label_field_idx] == "2":
                n_2 = n_2 + 1
    print('n0,n1,n2: %s,%s,%s' % (n_0, n_1, n_2))
    print('max sentence length: %s, mean sentence length: %s' % (max_sent_len, mean_sent_len // len(sentences)))
    return sentences, ['恶心呕吐'] * len(sentences), labels


def load_data():
    """
    加载恶心呕吐的模型
    """
    lines = []
    with open(input_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx > 0:
                lines.append(line)
    random.shuffle(lines)
    print('data rows: ', len(lines))
    max_sent_len, mean_sent_len = 0, 0
    sentences, labels = [], []
    row_field_num = len(lines[0].split(','))
    label_field_idx = 16
    n_0, n_1, n_2 = 0, 0, 0
    max_pos, mean_pos = 0, 0
    for line in lines:
        arr = line.strip().split(',')
        if len(arr) != row_field_num:
            print('illegal line: ' + line)
        else:
            sent_len = len(arr[1])
            if sent_len > max_sent_len:
                max_sent_len = sent_len
            mean_sent_len = mean_sent_len + sent_len

            if len(arr[1]) > 490:
                pos1 = max(arr[1].find('恶心'), arr[1].find('呕吐'))
                pos2 = max(arr[1].rfind('恶心'), arr[1].rfind('呕吐'))
                if pos2 - pos1 < 490:
                    sent = arr[1][max(pos1 - (pos2 - pos1) // 2, 0):490]
                else:
                    sent = arr[1][pos1:pos1+490]
                print(sent)
                sentences.append(sent)
            else:
                sentences.append(arr[1])

            labels.append(int(arr[label_field_idx]))
            if arr[label_field_idx] == "0":
                n_0 = n_0 + 1
            elif arr[label_field_idx] == "1":
                n_1 = n_1 + 1
            elif arr[label_field_idx] == "2":
                n_2 = n_2 + 1
    print('n0,n1,n2: %s,%s,%s' % (n_0, n_1, n_2))
    print('max sentence length: %s, mean sentence length: %s' % (max_sent_len, mean_sent_len // len(sentences)))
    return sentences, ['恶心呕吐'] * len(sentences), labels


def load_all_feature_data():
    """
    所有特征的训练数据
    """
    feature_cols = []
    lines = []
    sentences = []
    labels = None           # 样本标签
    labels_cls_count = []   # 所有特征样本每个类别的标签数量
    # 读取文件，存储特征名
    with open(input_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                feature_cols = line.strip().split('	')[4:29]
                labels = [[] for i in range(len(feature_cols))]
                labels_cls_count = [[0, 0, 0] for i in range(len(feature_cols))]
                print(feature_cols)
            else:
                arr = line.strip().split('	')
                lines.append('%s	%s' % (arr[1], '	'.join(arr[4:29])))
    random.shuffle(lines)
    print('data rows: ', len(lines))


    # 拆分训练数据
    for line in lines:
        arr = line.split('	')
        sentences.append(arr[0])
        for i,a in enumerate(arr[1:]):
            a_int = int(a)
            labels[i].append(a_int)
            labels_cls_count[i][a_int] = labels_cls_count[i][a_int] + 1

    # 特征名称训练对
    keywords_arr = [[feature] * len(lines) for feature in feature_cols]
    sentences_arr = [sentences for feature in feature_cols]
    print(len(sentences_arr), len(sentences_arr[0]), len(keywords_arr), len(keywords_arr[0]), len(labels), len(labels[0]), len(labels_cls_count), len(labels_cls_count[0]))

    return sentences_arr, keywords_arr, labels, labels_cls_count


def predict(model, data_loader, device):
    model.eval()
    results = []
    for batch in data_loader:
        input_ids, attention_mask, types, labels = [elem.to(device) for elem in batch]
        with torch.no_grad():
            pred = model(input_ids, attention_mask, types)
            pred = pred.detach().cpu().numpy()
            results.extend(np.argmax(pred, axis=1).flatten())

    return results


if __name__ == "__main__":
    #获取gpu和cpu的设备信息
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("use device: cuda")
        if torch.cuda.device_count() > 1:
            print("use multiply gpus: %s" % torch.cuda.device_count())
            # mymodel = torch.nn.DataParallel(mymodel)
    else:
        device = torch.device("cpu")
        print("use device: cpu")

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 数据
    # sentences, cmps, labels = load_data()
    # 加载6分类数据
    # sentences, features, labels, num_features = load_txt_data(id_map_6, input_file)
    # num_cls = 6
    # 循环加载数据

    # sentences_arr, keywords_arr, labels_arr, labels_cls_count_arr = load_all_feature_data()
    # # 单次训练
    # ct = 0
    # for sentences, cmps, labels, labels_cls_count in zip(sentences_arr, keywords_arr, labels_arr, labels_cls_count_arr):
    #     if ct == 1:
    #         writeToFile(result_file, 'feature %s training, sample nums: 0:%s, 1:%s, 2:%s' % (cmps[0], labels_cls_count[0], labels_cls_count[1], labels_cls_count[2]))
    #         print('feature %s training, sample nums: 0:%s, 1:%s, 2:%s' % (cmps[0], labels_cls_count[0], labels_cls_count[1], labels_cls_count[2]))
    #         datasets = DataToDataset(tokenizer, sentences, cmps, labels)
    #         train(datasets, device)
    #     ct = ct + 1

    # 所有数据混合训练
    # sentences_all, keywords_all, labels_all, labels_cls_count_all = [], [], [], [0, 0, 0]
    # for sentences, cmps, labels, labels_cls_count in zip(sentences_arr, keywords_arr, labels_arr, labels_cls_count_arr):
    #     sentences_all.extend(sentences)
    #     keywords_all.extend(cmps)
    #     labels_all.extend(labels)
    #     labels_cls_count_all = [labels_cls_count_all[i] + labels_cls_count[i] for i in range(3)]
    #
    # print('mix feature training, sample nums: 0:%s, 1:%s, 2:%s' % (labels_cls_count_all[0], labels_cls_count_all[1], labels_cls_count_all[2]))
    # writeToFile(result_file, 'mix feature training, sample nums: 0:%s, 1:%s, 2:%s' % (labels_cls_count_all[0], labels_cls_count_all[1], labels_cls_count_all[2]))
    # datasets = DataToDataset(tokenizer, sentences_all, keywords_all, labels_all)
    # train(datasets, device)

    # 使用模型预测
    # sentences_arr, keywords_arr, labels_arr, labels_cls_count_arr = load_all_feature_data()
    # model_names = ['9282', '7889', '9016', '9508', '9979', '9610', '8770', '9918', '9897', '8811',
    #                 '8668', '9938', '8934', '9651', '9467', '9569', '9077', '8196', '9303', '9877',
    #                 '9016', '9200', '9733', '9426', '9159']
    # for sentences, keywords, labels, model_name in zip(sentences_arr, keywords_arr, labels_arr, model_names):
    #     datasets = DataToDataset(tokenizer, sentences, keywords, labels, max_length)
    #     model_path = 'output/models/text/3_%s.pth' % model_name
    #     print('loading model: %s' % model_path)
    #     mymodel = torch.load(model_path)
    #     mymodel.to(device)
    #     data_loader = DataLoader(dataset=datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    #     results = predict(mymodel, data_loader, device)
    #     # 运行
    #     # 结果输出
    #     with open(r"output/textclassify_predict_20220513.txt", "a+") as f:
    #         for s, k, l, r in zip(sentences, keywords, labels, results):
    #             f.write('%s,%s,%s,%s\n' % (s, k, l, r))
