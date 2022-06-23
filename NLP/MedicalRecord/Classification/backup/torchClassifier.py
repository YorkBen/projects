# -*- coding:utf-8 -*-
# bert文本分类baseline模型
# model: bert
# date: 2021.10.10 10:01

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt


# 将数据构造成bert的输入格式
# inputs_ids: token的字典编码
# attention_mask:长度与inputs_ids一致，真实长度的位置填充1，padding位置填充0
# token_type_ids: 第一个句子填充0，第二个句子句子填充1
class MyDataset(Data.Dataset):
  def __init__(self, sentences, labels=None, with_labels=True,):
    self.tokenizer = BertTokenizer.from_pretrained(model_name)
    self.with_labels = with_labels
    self.sentences = sentences
    self.labels = labels

  def __len__(self):
    return len(sentences)

  def __getitem__(self, index):
    # Selecting sentence1 and sentence2 at the specified index in the data frame
    sent = self.sentences[index]

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    encoded_pair = self.tokenizer(sent,
                    padding = 'max_length',  # Pad to max_length
                    truncation = True,       # Truncate to max_length
                    max_length = max_length,
                    return_tensors = 'pt')  # Return torch.Tensor objects

    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    if self.with_labels:  # True if the dataset has labels
      label = self.labels[index]
      return token_ids, attn_masks, token_type_ids, label
    else:
      return token_ids, attn_masks, token_type_ids


# model
class BertClassify(nn.Module):
  def __init__(self):
    super(BertClassify, self).__init__()
    self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    self.linear = nn.Linear(hidden_size, n_class) # 直接用cls向量接全连接层分类
    self.dropout = nn.Dropout(0.5)

  def forward(self, X):
    input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
    # 用最后一层cls向量做分类
    # outputs.pooler_output: [bs, hidden_size]
    logits = self.linear(self.dropout(outputs.pooler_output))

    return logits




if __name__ == "__main__":
    # 参数
    batch_size = 16
    epoches = 10
    model_name = "hfl/chinese-roberta-wwm-ext-large"
    hidden_size = 768
    n_class = 8
    max_length = 256

    # 数据
    train_df = pd.read_csv(r'data\data_no.csv',delimiter='	',names=['text','label'])
    print('data shape: ', train_df.shape)
    sentences = list(train_df['text'])
    labels = train_df['label'].values
    train_loader = Data.DataLoader(dataset = MyDataset(sentences, labels), batch_size=batch_size, shuffle=True, num_workers=1)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bc = BertClassify().to(device)
    optimizer = optim.Adam(bc.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # 训练
    train_curve = []
    sum_loss = 0
    total_step = len(train_loader)
    for epoch in range(epoches):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = tuple(p.to(device) for p in batch)
            pred = bc([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, epoches, i+1, total_step, loss.item()))
        train_curve.append(sum_loss)
        sum_loss = 0

    pd.DataFrame(train_curve).plot() # loss曲线

    # 测试
    bc.eval()
    with torch.no_grad():
        test_text = ['我不喜欢打篮球']
        test = MyDataset(test_text, labels=None, with_labels=False)
        x = test.__getitem__(0)
        x = tuple(p.unsqueeze(0).to(device) for p in x)
        pred = bc([x[0], x[1], x[2]])
        pred = pred.data.max(dim=1, keepdim=True)[1]
        if pred[0][0] == 0:
            print('消极')
        else:
            print('积极')
