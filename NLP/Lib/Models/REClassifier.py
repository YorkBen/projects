import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics import accuracy_score
import logging
import datetime
import time

class DataToDataset(Dataset):
    def __init__(self, tokenizer, text_arr, labels, max_length):
        sentences_tokened = tokenizer(text_arr, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = sentences_tokened['input_ids']
        token_type_ids = sentences_tokened['token_type_ids']

        sharp_tokened = tokenizer(['#', '$', '&'], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        sharp_id, dollar_id, addr_id = sharp_tokened['input_ids'][0][1], sharp_tokened['input_ids'][1][1], sharp_tokened['input_ids'][2][1]
        print('#', sharp_id, '$', dollar_id, '&', addr_id)

        for r_idx, (input_id, token_type_id) in enumerate(zip(input_ids, token_type_ids)):
            i = 0
            while i < len(input_id):
                if input_id[i] == 0:
                    break
                if input_id[i] in [sharp_id, dollar_id, addr_id]:
                    # 匹配start
                    start = i
                    match, type = sharp_id, 1
                    if id == dollar_id:
                        match, type = dollar_id, 2
                    elif id == addr_id:
                        match, type = addr_id, 3

                    # 匹配end
                    i = i + 1
                    while i < len(input_id) and input_id[i] != match:
                        i = i + 1

                    if i == len(input_id):
                        break
                    else:
                        token_type_ids[r_idx][start:i+1] = type
                        input_ids[r_idx][start] = 0
                        input_ids[r_idx][i+1] = 0

                i = i+1

        self.input_ids = input_ids
        self.attention_mask = sentences_tokened['attention_mask']
        self.token_type_ids = token_type_ids
        self.labels = torch.tensor(labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        return self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.labels[index]

class BertTextClassficationModel(torch.nn.Module):
    def __init__(self, pre_model_path, hidden_size, num_cls, frozen_bert=False):
        super(BertTextClassficationModel, self).__init__()
        self.bert = BertModel.from_pretrained(pre_model_path)
        self.dense = torch.nn.Linear(hidden_size, num_cls)
        self.dropout = torch.nn.Dropout(0.5)
        self.frozen_bert = frozen_bert

    def forward(self, ids, mask, types):
        if self.frozen_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=types)
        else:
            outputs = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=types)

        logits = self.dense(self.dropout(outputs.pooler_output))
        return logits

class TextClassifier:
    def __init__(self, model_save_path='', pre_model_path="hfl/chinese-roberta-wwm-ext",
                    hidden_size=768, num_cls=2, max_txt_len=500, model_name='BertTextClassifier', model_file_path=None, device='cuda'):
        logging.info('Initializing Class TextClassifier...')
        # 预训练模型路径
        self.pre_model_path = pre_model_path
        # 隐藏层维度
        self.hidden_size = hidden_size
        # 分类类别数
        self.num_cls = num_cls
        # 文本最大长度
        self.max_txt_len = max_txt_len
        # 模型保存路径
        self.model_save_path = model_save_path
        # 模型保存名前缀
        self.model_name = model_name
        # 已经训练好的模型文件路径，如果该变量不为空，则直接加载该模型
        self.model_file_path = model_file_path

        # 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_model_path)

        # 加载模型
        if self.model_file_path is not None:
            self.model = torch.load(self.model_file_path)
            logging.debug('Loading Model %s' % self.model_file_path)
        else:
            self.model = BertTextClassficationModel(pre_model_path, hidden_size, num_cls)
            logging.debug('Initializing Model BertTextClassfication')

        # 初始化运行设备
        #获取gpu和cpu的设备信息
        if torch.cuda.is_available() and device == 'cuda':
            self.device = torch.device("cuda")
            self.model.to(self.device)
            logging.info("TextClassifier Using Device: CUDA")
            if torch.cuda.device_count() > 1:
                print("TextClassifier Using %s GPU" % torch.cuda.device_count())
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)
            logging.info("TextClassifier Using Device: CPU")


    def load_data(self, texts, labels, label_dict, texts_pair=None, is_training=True, train_ratio=0.8, batch_size=8):
        """
        加载数据：
            1. texts：文本数据
            2. texts_pair：文本数据对
            3. labels：标签数据。
            4. is_training：是否训练
            5. train_ratio：训练数据集占比，其余作为验证集
            6. batch_size：批处理尺寸
        输入数据行格式：
            1. 非文本对： text + separator + label
            2. 文本对： text1 + separator + text2 + separator + label
        """
        logging.info('TextClassifier Loading Data...')
        logging.debug('Using Text Pair? -> %s' % 'Yes' if texts_pair is not None else 'No')
        self.texts = texts
        self.texts_pair = texts_pair

        self.labels = labels
        # 标签转换未数字
        # label_elem_list = list(set(labels))
        self.label_num_dict = label_dict
        self.num_label_dict = {label_dict[k]:k for k in label_dict.keys()}
        labels = [self.label_num_dict[l] for l in labels]

        datasets = DataToDataset(self.tokenizer, texts, texts_pair, labels, self.max_txt_len)
        if is_training:
            train_size = int(len(datasets) * train_ratio)
            val_size = len(datasets) - train_size
            logging.debug("Data For Training, [train size, val size]: %s,%s" % (train_size, val_size))
            train_dataset, val_dataset = random_split(dataset=datasets, lengths=[train_size, val_size])
            self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            self.val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        else:
            logging.debug("Data For Predicting")
            self.predict_loader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=False, num_workers=1)



    def load_train_val_data(self, train_data, val_data, label_dict, batch_size=8):
        """
        加载数据：
            1. train_data：训练数据，(文本数据, 文本对, 分类)
            2. val_data：测试数据，(文本, 文本对, 分类)
            3. batch_size：批处理尺寸
        输入数据行格式：
            1. 非文本对： text + separator + label
            2. 文本对： text1 + separator + text2 + separator + label
        """
        logging.info('TextClassifier Loading Data...')
        logging.debug('Using Text Pair? -> %s' % 'Yes' if len(train_data[0]) == 3 else 'No')

        train_labels = [item[-1] for item in train_data]
        val_labels = [item[-1] for item in val_data]
        # 标签转换数字
        self.label_num_dict = label_dict
        self.num_label_dict = {label_dict[k]:k for k in label_dict.keys()}

        # 训练数据
        train_texts = [item[0] for item in train_data]
        # train_entity_pair = [((item[1], item[2], item[3]), (item[4], item[5], item[6])) for item in train_data] if len(train_data[0]) == 8 else None
        train_labels = [self.label_num_dict[l] for l in train_labels]
        train_dataset = DataToDataset(self.tokenizer, train_texts, train_labels, self.max_txt_len)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        # 测试数据
        val_texts = [item[0] for item in val_data]
        # val_texts_pair = [((item[1], item[2], item[3]), (item[4], item[5], item[6])) for item in val_data] if len(val_data[0]) == 8 else None
        val_labels = [self.label_num_dict[l] for l in val_labels]
        val_dataset = DataToDataset(self.tokenizer, val_texts, val_labels, self.max_txt_len)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)



    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds,axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat)


    def train(self, epochs=100, early_stopping_num=5, write_result_to_file=None):
        """
        训练
        """
        logging.debug("Start Model Traning....")
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-2)

        # 最好的验证acc， 当前未改进epoch数
        best_acc, no_improve_num, save_model_path = 0, 0, ''
        for epoch in range(epochs):
            # 训练 ######################
            start = time.time()
            train_loss, train_acc = 0.0, 0.0
            self.model.train()
            for data in self.train_loader:
                input_ids, attention_mask, types, labels = [elem.to(self.device) for elem in data]
                #优化器置零
                optimizer.zero_grad()
                #得到模型的结果
                out = self.model(input_ids, attention_mask, types)
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
                train_acc += self.flat_accuracy(out, labels)
            # epoch 指标计算
            train_loss = train_loss / len(self.train_loader)
            train_acc = train_acc / len(self.train_loader)
            end = time.time()
            logging.debug("Epoch: %d/%d, Training Loss:%f, Acc:%f, elapsed: %f" %(epoch, epochs, train_loss, train_acc, end-start))

            # 验证 #################################
            start = time.time()
            val_loss, val_acc = 0, 0
            self.model.eval()
            for data in self.val_loader:
                input_ids, attention_mask, types, labels = [elem.to(self.device) for elem in data]
                with torch.no_grad():
                    out = self.model(input_ids, attention_mask, types)
                    #计算误差
                    val_loss += loss_func(out, labels)
                    #计算acc
                    out = out.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    val_acc += self.flat_accuracy(out, labels)
            val_loss = val_loss / len(self.val_loader)
            val_acc = val_acc / len(self.val_loader)
            end = time.time()
            logging.debug("Evalaton loss:%f, Acc:%f, elapsed: %f" %(val_loss, val_acc, end-start))

            # 保存模型
            if val_acc > best_acc:
                no_improve_num = 0
                best_acc = val_acc
                save_model_path = r'%s/%s_%s_%s.pth' % (self.model_save_path, self.model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), int(best_acc * 10000))
                torch.save(self.model, save_model_path)
                logging.debug('Saving Model: %s' % save_model_path)
            else:
                no_improve_num = no_improve_num + 1
                if no_improve_num >= early_stopping_num:
                    logging.debug("No Improve more than %s epochs, Exit Training! Best accuracy: %f" % (no_improve_num, best_acc))
                    if write_result_to_file is not None:
                        with open(write_result_to_file, 'a+') as f:
                            f.write('%s, %s\n' % (self.model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
                            f.write('Best accuracy: %f\n' % best_acc)
                    return save_model_path


    def predict(self, write_result_to_file=None):
        """
        推理
        """
        results = []
        self.model.eval()
        for data in self.predict_loader:
            input_ids, attention_mask, types, labels = [elem.to(self.device) for elem in data]
            with torch.no_grad():
                pred = self.model(input_ids, attention_mask, types)
                pred = pred.detach().cpu().numpy()
                results.extend(np.argmax(pred, axis=1).flatten())

        # 结果输出
        with open(write_result_to_file, "a+") as f:
            if self.texts_pair is None:
                for k, e in enumerate(results):
                    f.write('%s,%s,%s\n' % (self.texts[k], self.labels[k], self.num_label_dict[e]))
            else:
                for k, e in enumerate(results):
                    f.write('%s,%s,%s,%s\n' % (self.texts[k], self.texts_pair[k], self.labels[k], self.num_label_dict[e]))


    def predict_nowrite(self, output_label=True):
        """
        推理
        """
        results, trans_labels = [], []
        self.model.eval()
        pred_acc = 0
        for data in self.predict_loader:
            input_ids, attention_mask, types, labels = [elem.to(self.device) for elem in data]
            with torch.no_grad():
                pred = self.model(input_ids, attention_mask, types)
                pred = pred.detach().cpu().numpy()
                labels = labels.cpu().detach().numpy()
                results.extend(np.argmax(pred, axis=1).flatten())
                trans_labels.extend(labels)
                pred_acc += self.flat_accuracy(pred, labels)
        pred_acc = pred_acc / len(self.predict_loader)

        print('best accuracy: %s' % pred_acc)
        print(self.num_label_dict)
        if output_label:
            return [(self.num_label_dict[r1], self.num_label_dict[r2]) for r1, r2 in zip(trans_labels, results)]
        else:
            return [(r1, r2) for r1, r2 in zip(trans_labels, results)]

#
# if __name__ == '__main__':
#     import random
#
#     def load_data(file_path):
#         """
#         输入数据格式：text \t relation name
#         """
#         data = []
#         with open(file_path) as f:
#             for line in f.readlines():
#                 arr = line.strip().split('\t')
#                 data.append((arr[0], arr[1]))
#
#         return data
#
#     def load_train_val_data(file_path):
#         data = load_data(file_path)
#
#         random.shuffle(data)
#         train_len = int(len(data) * 0.8)
#
#         train_data, val_data = data[:train_len], data[train_len:]
#         return train_data, val_data
#
#     train_data, val_data = load_train_val_data('train_gene_re_4c.txt')
#
#     label_cls_dict = {
#         'null': 0,
#         'negtive': 1,
#         'positive': 2,
#         'relatied': 3
#     }
#
#     # 初始化模型
#     model = TextClassifier(model_save_path='.',
#                             pre_model_path="bert-base-cased",
#                             # pre_model_path="../../Models/BertModels/gene",
#                             # pre_model_path="hfl/chinese-roberta-wwm-ext",
#                             num_cls=len(label_cls_dict),
#                             model_name='pre_gene_re_4c')
#
#     model.load_train_val_data(train_data, val_data, label_dict=label_cls_dict, batch_size=8)
