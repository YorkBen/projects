import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
import random
from sys import argv
from utils import *

torch.cuda.empty_cache()

# 参数
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext-large", 1024, r"models\torch\large", 1
model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext", 768, r"models\torch\common", 128
# model_name, hidden_size, model_output, BATCH_SIZE = "bert-base-chinese", 768, r"models\torch\google", 8

name = "feagen"
num_cls = 3
max_length = 500
epochs = 100
id_map = id_map_9

class BertTextClassficationModel(torch.nn.Module):
    def __init__(self):
        super(BertTextClassficationModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dense1 = torch.nn.Linear(hidden_size, 1024)
        self.dense2 = torch.nn.Linear(1024, num_cls)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, ids, mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=ids, attention_mask=mask)
        # outputs = self.bert(input_ids=ids, attention_mask=mask)
        x = F.relu(self.dense1(outputs.pooler_output))
        x = self.dense2(x)
        return x


def train(datasets):
    print('Training model...')
    train_size = int(len(datasets) * 0.8)
    val_size = len(datasets) - train_size
    print("[train size, test size]: ", [train_size, val_size])

    train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # 模型
    mymodel = BertTextClassficationModel()
    device = choose_device(mymodel)
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
            input_ids, attention_mask, labels = [elem.to(device) for elem in data]
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
            out = out.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_acc += flat_accuracy(out, labels)

        end = time.time()
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        print("Epoch: %d/%d, Training Loss:%f, Acc:%f, elapsed: %f" %(epoch, epochs, train_loss, train_acc, end-start))


        val_loss = 0
        val_acc = 0
        mymodel.eval()
        start = time.time()
        for j, batch in enumerate(val_loader):
            val_ids, val_mask, val_labels = [elem.to(device) for elem in batch]
            with torch.no_grad():
                pred = mymodel(val_ids, val_mask)
                val_loss += loss_func(pred, val_labels)
                pred = pred.detach().cpu().numpy()
                val_labels = val_labels.detach().cpu().numpy()
                val_acc += flat_accuracy(pred, val_labels)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        end = time.time()
        print("\t      Evalaton loss:%f, Acc:%f, elapsed: %f" %(val_loss, val_acc, end-start))

        if val_acc > best_acc1:
            no_improve_num = 0
            best_acc1 = val_acc
            if best_acc1 > 0.9:
                best_modelname = r'models/%s/%s_%s.pth' % (name, num_cls, int(best_acc1 * 10000))
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
    # 厌食,恶心呕吐,右下腹疼痛,转移性右下腹疼痛  => 0,1,2,3
    idx = 1
    str = ''.join(np.array(features, dtype=str)[:, idx].tolist())
    n_0 = str.count('0')
    n_1 = str.count('1')
    n_2 = str.count('2')

    print('0vs1vs2 sample num: %s:%s:%s' % (n_0, n_1, n_2))
    input_ids, attention_mask = text_tokenize(model_name, sentences, max_length)

    # 封装数据
    # datasets = TextDataToDataset(input_ids, attention_mask, labels)
    datasets = TextLabelToDataset(input_ids, attention_mask, features, idx)
    # datasets = FeatureDataToDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels))

    if len(argv) >= 2 and argv[1] == "inference":
        inference(datasets, sentences, features, labels)
    else:
        train(datasets)
