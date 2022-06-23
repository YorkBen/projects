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
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext-large", 1024, r"models\torch\large", 1
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext", 768, r"output/models/clss/common", 8
model_name, hidden_size, BATCH_SIZE = "BertModels/medical-roberta-wwm", 768, 8
# model_name, hidden_size, model_output, BATCH_SIZE = "BertModels/test-mlm", 768, r"output/models/clss/common", 8

# input_file, num_cls = r'data/processed2.txt', 2    # 急性阑尾炎
input_file, num_cls = r'data/train_data_20220424.txt', 3
max_length = 500
epochs = 100
model_dir = 'text'


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
        # with torch.no_grad():
        #     outputs = self.bert(input_ids=ids, attention_mask=mask)
        outputs = self.bert(input_ids=ids, attention_mask=mask)
        logits = self.dense(self.dropout(outputs.pooler_output))
        return logits

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def load_data():
    lines = []
    with open(input_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx > 0:
                lines.append(line)
    random.shuffle(lines)
    print('data rows: ', len(lines))
    sentences, labels = [], []
    row_field_num = len(lines[0].split(','))
    label_field_idx = 16
    n_0, n_1, n_2 = 0, 0, 0
    for line in lines:
        arr = line.strip().split(',')
        if len(arr) != row_field_num:
            print('illegal line: ' + line)
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

    return sentences, labels


if __name__ == "__main__":
    # 数据
    sentences, labels = load_data()
    # 加载6分类数据
    # sentences, features, labels, num_features = load_txt_data(id_map_6, input_file)
    # num_cls = 6

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
        mymodel = torch.load(r"output/models/%s/6_8800.pth" % model_dir)
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
        with open(r"output/models/%s/6_8800.txt" % model_dir, "w") as f:
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
                # # 调试
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
                best_modelname = r'output/models/%s/%s_%s.pth' % (model_dir, num_cls, int(best_acc1 * 10000))
                torch.save(mymodel, best_modelname)
            else:
                no_improve_num = no_improve_num + 1
                if no_improve_num > 5:
                    print("no improve more than: %s, exit training. best accuracy: %f" % (no_improve_num, best_acc1))
                    exit()
