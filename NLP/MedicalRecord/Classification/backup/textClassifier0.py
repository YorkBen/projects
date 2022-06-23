import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
import random
from sys import argv

torch.cuda.empty_cache()

# 参数
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext-large", 1024, r"models\torch\large", 1
model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext", 768, r"models\torch\common", 32
# model_name, hidden_size, model_output, BATCH_SIZE = "bert-base-chinese", 768, r"models\torch\google", 8

# model_name = "hfl/chinese-roberta-wwm-ext-large"
# input_file, num_cls = r'data\data_no0.csv', 9      # 测试数据
# input_file, num_cls = r'data\data_no_9.csv', 9      # 急性 + 慢性
# input_file, num_cls = r'data\data_no_8.csv', 8      # 急性 + 慢性 去掉GA18
input_file, num_cls = r'data\data_no_6.csv', 6    # 急性阑尾炎
max_length = 500
epochs = 100

id_map_6 = ["GB70", "DC31", "DB10", "DA91", "JA01", "DC11"]


class DataToDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

class BertTextClassficationModel(torch.nn.Module):
    def __init__(self):
        super(BertTextClassficationModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dense = torch.nn.Linear(hidden_size, num_cls)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, ids, mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=ids, attention_mask=mask)
        # outputs = self.bert(input_ids=ids, attention_mask=mask)
        logits = self.dense(self.dropout(outputs.pooler_output))
        return logits

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


if __name__ == "__main__":
    # 数据
    lines = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            lines.append(line)
    random.shuffle(lines)
    print('data rows: ', len(lines))
    sentences, labels = [], []
    for line in lines:
        arr = line.split('	')
        if len(arr) != 2:
            print('illegal line: ' + line)
        else:
            sentences.append(arr[0])
            labels.append(int(arr[1]))

    tokenizer = BertTokenizer.from_pretrained(model_name)
    sentences_tokened = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    input_ids, attention_mask = sentences_tokened['input_ids'], sentences_tokened['attention_mask']
    print("sentences tokenize finished...")
    labels = torch.tensor(labels)


    #获取gpu和cpu的设备信息
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("use device: cuda")
        if torch.cuda.device_count() > 1:
            print("use multiply gpus: %s" % torch.cuda.device_count())
            mymodel = torch.nn.DataParallel(mymodel)
    else:
        device = torch.device("cpu")
        print("use device: cpu")

    # 封装数据
    datasets = DataToDataset(input_ids, attention_mask, labels)

    if len(argv) >= 2 and argv[1] == "test":
        print('testing model...')
        mymodel = torch.load(r"models\torch\common\6_8800.pth")
        mymodel.to(device)
        data_loader = DataLoader(dataset=datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        # 运行
        mymodel.eval()
        results = []
        for j, batch in enumerate(data_loader):
            test_input_ids, test_attention_mask, val_labels = [elem.to(device) for elem in batch]
            with torch.no_grad():
                pred = mymodel(test_input_ids, test_attention_mask)
                pred = pred.detach().cpu().numpy()
                results.extend(np.argmax(pred, axis=1).flatten())

        # 结果输出
        with open(r"models\torch\common\6_8800.txt", "w") as f:
            for sentence, label, r in zip(sentences, labels, results):
                f.write('%s	%s	%s\n' % (sentence, id_map_6[label], id_map_6[r]))


    else:
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
                input_ids, attention_mask, labels = [elem.to(device) for elem in data]
                # 调试
                if print_num == 0:
                    print(torch.cuda.memory_stats())
                    print_num = 1

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
                val_input_ids, val_attention_mask, val_labels = [elem.to(device) for elem in batch]
                with torch.no_grad():
                    pred = mymodel(val_input_ids, val_attention_mask)
                    val_loss += loss_func(pred,val_labels)
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
                best_modelname = r'%s\%s_%s.pth' % (model_output, num_cls, int(best_acc1 * 10000))
                torch.save(mymodel, best_modelname)
            else:
                no_improve_num = no_improve_num + 1
                if no_improve_num > 5:
                    print("no improve more than: %s, exit training. best accuracy: %f" % (no_improve_num, best_acc1))
                    exit()
