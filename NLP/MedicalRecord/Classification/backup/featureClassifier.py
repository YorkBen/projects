import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
from sys import argv
from utils import *

torch.cuda.empty_cache()

# 参数
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext-large", 1024, r"models\torch\large", 1
# model_name, hidden_size, model_output, BATCH_SIZE = "hfl/chinese-roberta-wwm-ext", 768, r"models\torch\common", 32
# model_name, hidden_size, model_output, BATCH_SIZE = "bert-base-chinese", 768, r"models\torch\google", 8

# model_name = "hfl/chinese-roberta-wwm-ext-large"
# input_file, num_cls = r'data\data_no0.csv', 9      # 测试数据
# input_file, num_cls = r'data\data_no_9.csv', 9      # 急性 + 慢性
# input_file, num_cls = r'data\data_no_8.csv', 8      # 急性 + 慢性 去掉GA18
# input_file, num_cls = r'data\data_no_6.csv', 6    # 急性阑尾炎
max_length = 500
epochs = 100
# feature_size = 29
hidden_size1 = 512
hidden_size2 = 512
BATCH_SIZE = 256
name = "feature"

id_map, num_cls = id_map_6, 6
# idc11_id, num_cls = "DB10", 2


class FeatureClassficationModel(torch.nn.Module):
    def __init__(self, feature_size):
        super(FeatureClassficationModel, self).__init__()
        self.dense1 = torch.nn.Linear(feature_size, hidden_size1)
        self.dense2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.dense3 = torch.nn.Linear(hidden_size2, num_cls)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


def train(datasets, feature_size):
    print('Training model...')
    train_size = int(len(datasets) * 0.8)
    val_size = len(datasets) - train_size
    print("[train size, test size]: ", [train_size, val_size])

    train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # 模型
    mymodel = FeatureClassficationModel(feature_size)
    # device = choose_device(mymodel)
    device = 'cuda'
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
            features, labels = [elem.to(device) for elem in data]
            # # 调试
            # if print_num == 0:
            #     print(torch.cuda.memory_stats())
            #     print_num = 1

            #优化器置零
            optimizer.zero_grad()
            #得到模型的结果
            out = mymodel(features)
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
            val_features, val_labels = [elem.to(device) for elem in batch]
            with torch.no_grad():
                pred = mymodel(val_features)
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
            if best_acc1 > 0.7:
                best_modelname = r'models/%s/%s_%s.pth' % (name, num_cls, int(best_acc1 * 10000))
                torch.save(mymodel, best_modelname)
        else:
            no_improve_num = no_improve_num + 1
            if no_improve_num > 5:
                print("no improve more than: %s, exit training. best accuracy: %f" % (no_improve_num, best_acc1))
                exit()


def inference(datasets, sentences, features, labels):
    print('Inference model...')
    mymodel = torch.load(r"models/%s/6_7437.pth" % name)
    device = choose_device(mymodel)
    mymodel.to(device)
    data_loader = DataLoader(dataset=datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # 运行
    mymodel.eval()
    results = []
    for j, batch in enumerate(data_loader):
        val_features, _ = [elem.to(device) for elem in batch]
        with torch.no_grad():
            pred = mymodel(val_features)
            pred = pred.detach().cpu().numpy()
            results.extend(np.argmax(pred, axis=1).flatten())

    # 结果输出
    with open(r"models/%s/6_7437.txt" % name, "w") as f:
        for sentence, label, r in zip(sentences, labels, results):
            f.write('%s	%s	%s\n' % (sentence, id_map[label], id_map[r]))


if __name__ == "__main__":
    # 数据
    sentences, features, labels, feature_size = load_data(id_map = id_map, input_file=r'data/train_data_20220424.txt')
    # sentences, features, labels = load_data_one_id(id = idc11_id)
    # input_ids, attention_mask = text_tokenize(sentences)

    # 封装数据
    # datasets = TextDataToDataset(input_ids, attention_mask, labels)
    # datasets = MixDataToDataset(input_ids, attention_mask, features, labels)
    datasets = FeatureDataToDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels))

    if len(argv) >= 2 and argv[1] == "inference":
        inference(datasets, sentences, features, labels)
    else:
        train(datasets, feature_size)
