import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import multilabel_confusion_matrix as mcm, classification_report
import time
import random
from sys import argv
from utils import *

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

name = "feagen"
id_map = id_map_6
feature_size = 4
feature_size = 1
max_length = 500
epochs = 100


class BertFeatureExtractionModel(torch.nn.Module):
    def __init__(self):
        super(BertFeatureExtractionModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dense1 = torch.nn.Linear(hidden_size, 512)
        self.dense2 = torch.nn.Linear(512, feature_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ids, mask):
        # 此处是否冻结Bert的层来训练
        with torch.no_grad():
            outputs = self.bert(input_ids=ids, attention_mask=mask)
        # outputs = self.bert(input_ids=ids, attention_mask=mask)
        # x = self.dense(self.dropout(outputs.pooler_output))
        x = F.relu(self.dense1(outputs.pooler_output))
        x = self.dense2(x)

        return self.sigmoid(x)


def flat_accuracy(preds, labels):
    correct = sum(row.all().int().item() for row in (preds.ge(0.5) == labels))
    n = labels.shape[0]
    return correct / n


def train(datasets, idx):
    print('Training model...')
    train_size = int(len(datasets) * 0.8)
    val_size = len(datasets) - train_size
    print("[train size, test size]: ", [train_size, val_size])

    train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # 模型
    mymodel = BertFeatureExtractionModel()
    device = choose_device(mymodel)
    mymodel.to(device)
    loss_func = torch.nn.BCELoss()
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
            input_ids, attention_mask, features, _ = [elem.to(device) for elem in data]
            # labels = features[:, 4:]
            labels = features[:, idx].unsqueeze(-1)
            # 调试
            # if print_num == 0:
            #     print(torch.cuda.memory_stats())
            #     print_num = 1

            #优化器置零
            optimizer.zero_grad()
            #得到模型的结果
            out = mymodel(input_ids, attention_mask)
            #计算误差
            loss = loss_func(out, labels)
            train_loss += loss.item()
            #误差反向传播
            loss.backward()
            #更新模型参数
            optimizer.step()
            #计算acc
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
            val_ids, val_mask, val_features, _ = [elem.to(device) for elem in batch]
            # val_labels = val_features[:, 4:]
            val_labels = val_features[:, idx].unsqueeze(-1)
            with torch.no_grad():
                pred = mymodel(val_ids, val_mask)
                val_loss += loss_func(pred, val_labels)
                val_acc += flat_accuracy(pred, val_labels)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        end = time.time()
        print("\t      Evalaton loss:%f, Acc:%f, elapsed: %f" %(val_loss, val_acc, end-start))

        if val_acc > best_acc1:
            no_improve_num = 0
            best_acc1 = val_acc
            best_modelname = r'models/%s/%s_%s.pth' % (name, feature_size, int(best_acc1 * 10000))
            torch.save(mymodel, best_modelname)
        else:
            no_improve_num = no_improve_num + 1
            if no_improve_num > 5:
                print("no improve more than: %s, exit training. best accuracy: %f" % (no_improve_num, best_acc1))
                exit()


def inference(datasets, sentences, features, labels):
    print('Inference model...')
    mymodel = torch.load(r"models/%s/common/6_8800.pth" % name)
    device = choose_device(mymodel)
    mymodel.to(device)
    data_loader = DataLoader(dataset=datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # 运行
    mymodel.eval()
    results = []
    for j, batch in enumerate(data_loader):
        val_ids, val_mask, val_features, _ = [elem.to(device) for elem in batch]
        with torch.no_grad():
            pred = mymodel(val_ids, val_mask)
            pred = pred.detach().cpu().numpy()
            results.extend(np.argmax(pred, axis=1).flatten())

    # 结果输出
    with open(r"models/%s/common/6_8800.txt", "w") as f:
        for sentence, label, r in zip(sentences, labels, results):
            f.write('%s	%s	%s\n' % (sentence, id_map_6[label], id_map_6[r]))


if __name__ == "__main__":
    # 数据
    sentences, features, labels = load_data(id_map = id_map)
    idx = 5
    p_count = np.sum(np.array(features)[:, idx])
    n_count = len(features) - p_count
    print('positive vs total sample num: %s:%s' % (p_count, n_count))
    input_ids, attention_mask = text_tokenize(model_name, sentences, max_length)

    # 封装数据
    # datasets = TextDataToDataset(input_ids, attention_mask, labels)
    datasets = MixDataToDataset(input_ids, attention_mask, torch.tensor(features, dtype=torch.float32), torch.tensor(labels))
    # datasets = FeatureDataToDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels))

    if len(argv) >= 2 and argv[1] == "inference":
        inference(datasets, sentences, features, labels)
    else:
        train(datasets, idx)
