import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

def load_data_file(file_path, feature_num):
    """
    加载数据文件，文件格式：
    f1, f2, f3, ... fn, lbl1, lbl2, lbl3...
    """
    features, labels = [], []
    with open(file_path) as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            features.append([float(e) for e in arr[1:feature_num+1]])
            labels_arr = [float(1) if float(e) > 1 else float(e) for e in arr[feature_num+1:]]
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return self.sigmoid(x)


def train(dataloader, mymodel, loss_func, optimizer):
    size = len(dataloader.dataset)
    mymodel.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = mymodel(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = loss.item()
    print(f"Train Loss: {loss:>7f}")

def calc_acc(pred, y):
    pred = (pred > 0.5).float().detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    c_num, c_total = 0, 0
    fp, fn, tp, tn = 0, 0, 0, 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j] == 1 and pred[i][j] == 1:
                tp = tp + 1
            elif y[i][j] == 0 and pred[i][j] == 1:
                fp = fp + 1
            elif y[i][j] == 0 and pred[i][j] == 0:
                tn = tn + 1
            elif y[i][j] == 1 and pred[i][j] == 0:
                fn = fn + 1

    precision = (tp) / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # print(fp, fn, tp, tn, precision, recall, f1)

    return f1


def val(dataloader, mymodel, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mymodel.eval()
    test_loss, f1_score = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = mymodel(X)
            test_loss += loss_func(pred, y).item()

            f1_score = f1_score + calc_acc(pred, y)
            # print(f1_score)

    test_loss /= num_batches
    f1_score /= num_batches
    print(f"Test Loss: {test_loss:>8f}, Test F1 Score: {f1_score*100:>8.2f}% \n")
    return test_loss, f1_score


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
    n_epochs = 100
    BATCH_SIZE = 16
    ni_tolerance = 5
    n_features, n_labels, n_hidden = 110, 11, 64

    # 加载数据
    features, labels = load_data_file(r'data\train_data_202207.txt', n_features)
    train_size = int(len(labels) * 0.8)
    val_size = len(labels) - train_size
    print('train data size: %s, val data size: %s' % (train_size, val_size))
    print('train feature num: %s, predict label num: %s' % (len(features[0]), len(labels[0])))
    datasets = DataToDataset(features, labels)
    train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # 训练
    mymodel = Classifier(n_features, n_hidden, n_labels)
    mymodel.to(device)

    loss_func = nn.BCELoss()
    # loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.0001, weight_decay=1e-2)
    # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    # warmup_proportion = 0.1
    # warmup_steps = int(n_epochs * warmup_proportion)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_epochs)


    best_acc, no_improve_num = 0, 0
    for epoch in range(n_epochs):
        print('Epoch: %s/%s' % (epoch, n_epochs))
        train(train_loader, mymodel, loss_func, optimizer)
        val_loss, val_acc = val(val_loader, mymodel, loss_func)
        if val_acc > best_acc:
            print('saving model: %s_%s.pth' % (datetime.now().strftime("%Y%m%d"), val_acc))
            torch.save(mymodel, r'output\models\multilabel\%s_%s.pth' % (datetime.now().strftime("%Y%m%d"), val_acc))
            best_acc, no_improve_num = val_acc, 0
        else:
            no_improve_num = no_improve_num + 1
            print('no improve num: %s' % no_improve_num)

        if no_improve_num >= ni_tolerance:
            break


#
