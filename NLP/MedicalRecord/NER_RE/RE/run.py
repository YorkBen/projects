import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
import argparse
import random
from utils import *

torch.cuda.empty_cache()

# 参数
model_name, hidden_size, model_output, BATCH_SIZE = "../../BertModels/medical-roberta-wwm", 768, 'output', 8

name = "rebert"
num_cls = 4
max_length = 500
epochs = 50

class BertTextClassficationModel(torch.nn.Module):
    def __init__(self):
        super(BertTextClassficationModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.dense3 = torch.nn.Linear(hidden_size*3, num_cls)

    def forward(self, ids, mask, pos_pair):
        with torch.no_grad():
            # outputs = self.bert(input_ids=ids, attention_mask=mask)
            output = self.bert(input_ids=ids, attention_mask=mask)
            sequence_output = output.last_hidden_state
            pooler_output = output.pooler_output
            # print(sequence_output.shape, pooler_output.shape)

        # batch_size
        token_embdings_1, token_embdings_2 = [], []
        for k in range(len(pos_pair)):
            pos_pair_item = pos_pair[k]

            # token_1，为起止标识符的位置
            pos_1 = pos_pair_item[0]
            token_embdings_1.append(sequence_output[k][int(pos_1[0])+1:int(pos_1[1])].mean(axis=0).unsqueeze(dim=0))

            # token_2
            pos_2 = pos_pair_item[1]
            token_embdings_2.append(sequence_output[k][int(pos_2[0])+1:int(pos_2[1])].mean(axis=0).unsqueeze(dim=0))


        x1 = self.dense1(self.dropout(F.relu(torch.cat(token_embdings_1)))) # torch.cat 上下拼接
        x2 = self.dense1(self.dropout(F.relu(torch.cat(token_embdings_2)))) # torch.cat 上下拼接

        # pooling layer
        x3 = self.dense2(self.dropout(F.relu(pooler_output)))

        # concat
        x = torch.cat([x1, x2, x3], axis=1)

        x = self.dense3(self.dropout(x))
        return x


def train(datasets):
    print('Training model...')
    train_size = int(len(datasets) * 0.8)
    val_size = len(datasets) - train_size
    train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])

    # print("[train size, test size]: ", [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # 模型
    mymodel = BertTextClassficationModel()
    device = choose_device(mymodel)
    mymodel.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.00002, weight_decay=1e-2)

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
            input_ids, attention_mask, pos_pair, labels = [elem.to(device) for elem in data]
            # 调试
            # if print_num == 0:
            #     print(torch.cuda.memory_stats())
            #     print_num = 1

            #优化器置零
            optimizer.zero_grad()
            #得到模型的结果
            out = mymodel(input_ids, attention_mask, pos_pair)
            #计算误差
            # labels = labels.to(torch.int64)
            # print(out.dtype)
            # print(labels.dtype)

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
            val_ids, val_mask, pos_pair, val_labels = [elem.to(device) for elem in batch]
            with torch.no_grad():
                pred = mymodel(val_ids, val_mask, pos_pair)
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
                best_modelname = r'output/%s_%s.pth' % (num_cls, int(best_acc1 * 10000))
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
        val_ids, val_mask, pos_pair, _ = [elem.to(device) for elem in batch]
        with torch.no_grad():
            pred = mymodel(val_ids, val_mask, pos_pair)
            pred = pred.detach().cpu().numpy()
            results.extend(np.argmax(pred, axis=1).flatten())

    # 结果输出
    with open(r"models/%s/common/6_8800.txt", "w") as f:
        for sentence, label, r in zip(sentences, labels, results):
            f.write('%s	%s	%s\n' % (sentence, id_map_6[label], id_map_6[r]))


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
    parser.add_argument('-i', type=str, default='project.json', help='input file')
    parser.add_argument('-o', type=str, default='train_data.txt', help='output file')
    parser.add_argument('-t', type=str, default='train', help='data type')
    args = parser.parse_args()

    input = args.i
    output = args.o
    type = args.t

    print("input: %s, output: %s, type: %s" % (input, output, type))
    if type not in ['train', 'inference']:
        print('Error: parameter type must be one of [train, inference]')
        exit()

    sentences, pos_pair, labels = load_data(input)
    input_ids, attention_mask = text_tokenize(model_name, sentences, max_length)
    datasets = TextLabelToDataset(input_ids, attention_mask, pos_pair, labels)

    if type == "inference":
        inference(datasets, sentences, features, labels)
    else:
        train(datasets)
