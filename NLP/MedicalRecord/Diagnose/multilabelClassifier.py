import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def load_data_file(file_path, n_labels, separator='	', skip_head=True):
    """
    加载数据文件，文件格式：
    f1, f2, f3, ... fn, lbl1, lbl2, lbl3...
    """
    features, labels = [], []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and skip_head:
                continue
            arr = line.strip().split(separator)
            features.append([float(e) for e in arr[:-n_labels]])
            labels_arr = [float(1) if float(e) > 1 else float(e) for e in arr[-n_labels:]]
            labels.append(labels_arr)

    return features, labels


class DataToDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return torch.FloatTensor(self.features[index]), torch.FloatTensor(self.labels[index])


class Classifier(nn.Module):
    def __init__(self, n_features, n_hidden, n_class):
        super(Classifier, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class)
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.linear_relu_stack(x)
        # return self.sigmoid(x)
        return self.linear_relu_stack(x)

class Evaluater():
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.record_num = 0
        self.correct_record_num = 0

    def compute(self, y, preds):
        # preds = (preds > 0.5).float().detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        preds = (np.array(preds) >= 0.5).astype(int)
        for r1, r2 in zip(y, preds):
            self.record_num = self.record_num + 1
            has_diff = False
            for e1, e2 in zip(r1, r2):
                if e1 == 1 and e2 == 1:
                    self.tp = self.tp + 1
                elif e1 == 1 and e2 == 0:
                    self.fn = self.fn + 1
                    has_diff = True
                elif e1 == 0 and e2 == 0:
                    self.tn = self.tn + 1
                else:
                    self.fp = self.fp + 1
                    has_diff = True

            if not has_diff:
                self.correct_record_num = self.correct_record_num + 1


    def calc(self):
        precision = float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0
        recall = float(self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else 0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        record_acc = float(self.correct_record_num / self.record_num)
        return precision, recall, f1, record_acc



def train(dataloader, mymodel, loss_func, optimizer):
    size = len(dataloader.dataset)
    mymodel.train()
    train_loss = 0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = mymodel(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
    print(f"Train Loss: {train_loss:>7f}")


def val(dataloader, mymodel, loss_func):
    size = len(dataloader.dataset)
    mymodel.eval()
    valid_loss = 0

    evaluator = Evaluater()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = mymodel(X)
            loss = loss_func(pred, y)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            # fp, fn, tp, tn = calc_acc(pred, y)
            # fp_all = fp_all + fp
            # fn_all = fn_all + fn
            # tp_all = tp_all + tp
            # tn_all = tn_all + tn
            # evaluator.compute(y, pred)
            # print(f1_score)
            val_targets.extend(y.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(pred).cpu().detach().numpy().tolist())

    evaluator.compute(val_targets, val_outputs)
    precision, recall, f1, record_acc = evaluator.calc()
    print(f"Evaluation Loss: {valid_loss:>8f}, Precision: { precision*100:>4.2f}%, Recall: {recall *100:>4.2f}%, F1: {f1 *100:>4.2f}%, Record Acc: {record_acc *100:>4.2f}% \n")

    # 模型预测指标
    val_predicts = (np.array(val_outputs) >= 0.5).astype(int)
    accuracy = accuracy_score(val_targets, val_predicts)
    f1_score_micro = f1_score(val_targets, val_predicts, average='micro')
    f1_score_macro = f1_score(val_targets, val_predicts, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    # print(classification_report(val_targets, val_predicts))

    return valid_loss, precision, recall, f1, record_acc


if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("use device: cuda")
        if torch.cuda.device_count() > 1:
            print("use multiply gpus: %s" % torch.cuda.device_count())
            # mymodel = torch.nn.DataParallel(mymodel)
    else:
        device = torch.device("cpu")
        print("use device: cpu")

    # 参数设置
    n_epochs = 200
    BATCH_SIZE = 128
    ni_tolerance = 30
    n_labels, n_hidden = 10, 64

    # 加载数据
    train_X, train_y = load_data_file(r'data\训练_全特征_多诊断.txt', n_labels)
    val_X, val_y = load_data_file(r'data\测试_全特征_多诊断.txt', n_labels)
    n_features = len(train_X[0])

    # train_size = int(len(labels) * 0.8)
    # val_size = len(labels) - train_size
    # print('train data size: %s, val data size: %s' % (train_size, val_size))
    # print('train feature num: %s, predict label num: %s' % (len(features[0]), len(labels[0])))
    # datasets = DataToDataset(features, labels)
    # train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    train_loader = DataLoader(dataset=DataToDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=DataToDataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)



    # 训练
    mymodel = Classifier(n_features, n_hidden, n_labels)
    mymodel.to(device)

    # loss_func = nn.BCELoss()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.0001, weight_decay=1e-2)
    # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    # warmup_proportion = 0.1
    # warmup_steps = int(n_epochs * warmup_proportion)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_epochs)

    best_loss, best_f1, no_improve_num = 1000000, 0, 0
    for epoch in range(n_epochs):
        print('Epoch: %s/%s' % (epoch, n_epochs))
        val_targets = []
        val_outputs = []
        train(train_loader, mymodel, loss_func, optimizer)
        val_loss, val_prec, val_recall, val_f1, record_acc = val(val_loader, mymodel, loss_func)
        if val_loss < best_loss:
            print('saving model: %s_%s.pth' % (datetime.now().strftime("%Y%m%d"), val_loss))
            torch.save(mymodel, r'output\models\multilabel\%s_%.4f.pth' % (datetime.now().strftime("%Y%m%d"), val_loss))
            best_loss, no_improve_num = val_loss, 0
        # if val_f1 > best_f1:
        #     save_mdl_name = '%s_%s.pth' % (datetime.now().strftime("%Y%m%d"), val_f1)
        #     print('saving model: %s' % save_mdl_name)
        #     torch.save(mymodel, r'output\models\multilabel\%s' % save_mdl_name)
        #     best_f1, no_improve_num = val_f1, 0
        else:
            no_improve_num = no_improve_num + 1
            print('no improve num: %s' % no_improve_num)

        if no_improve_num >= ni_tolerance:
            print('best loss: %s' % best_loss)
            break




#
