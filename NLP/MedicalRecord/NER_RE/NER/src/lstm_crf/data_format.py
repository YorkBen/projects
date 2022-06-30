# -*- coding:utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AlbertConfig, BertTokenizer, AlbertModel
# from transformers import BertTokenizer, BertModel


import os

class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask, output_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask
        self.output_mask = output_mask

class DataFormat():
    def __init__(self, max_length=100, batch_size=20, data_type='train'):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.train_data = []
        self.tag_map = {'B_Symptom': 0,
                   'I_Symptom': 1,
                   'B_Physiology': 2,
                   'I_Physiology': 3,
                   'B_Status': 4,
                   'I_Status': 5,
                   'B_Property': 6,
                   'I_Property': 7,
                   'B_Body': 8,
                   'I_Body': 9,
                   'B_Value': 10,
                   'I_Value': 11,
                        'O': 12}
        base_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if data_type == "train":
            self.data_path = base_path + '/data/ner_data/train/'
        elif data_type == "dev":
            self.data_path = base_path + "/data/ner_data/dev/"
        elif data_type == "test":
            self.data_path = base_path + "/data/ner_data/test/"

        self.read_corpus(self.data_path + 'source.txt', self.data_path + 'target.txt', self.max_length, self.tag_map)
        self.train_dataloader= self.prepare_batch(self.train_data, self.batch_size)


    def read_corpus(self, train_file_data, train_file_tag, max_length, label_dic):
        """
        :param train_file_data:训练数据
        :param train_file_tag: 训练数据对应的标签
        :param max_length: 训练数据每行的最大长度
        :param label_dic: 标签对应的索引
        :return:
        """
        #因为中文没有sentencepiece，故通过hugging face的bert_tokenize进行词索引的转换
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # tokenizer = BertTokenizer.from_pretrained('BertModels/medical-roberta-wwm')

        with open(train_file_data, 'r', encoding='utf-8') as file_train:
            with open(train_file_tag, 'r', encoding='utf-8') as file_tag:
                train_data = file_train.readlines()
                tag_data = file_tag.readlines()
                for text, label in zip(train_data, tag_data): #每个字对应每个实体
                    tokens = text.split()
                    label = label.split()
                    if len(tokens) > max_length - 2:  # 大于最大长度进行截断 并保留cls和sep
                        tokens = tokens[0:(max_length - 2)]
                        label = label[0:(max_length - 2)]
                    label_cs = ' '.join(label)

                    #自动加上cls和sep
                    input_ids = tokenizer.encode(tokens, add_special_tokens=True)
                    # tag -> index 这里没有对label_ids加[CLS]和[SEP]
                    label_ids = [label_dic[i] for i in label_cs.split()]
                    input_mask = [1] * len(input_ids)

                    #当句子长度小于max_length时进行padding 0操作
                    while len(input_ids) < max_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        #pad（0）对应的标签为-1
                    while len(label_ids) < max_length:
                        label_ids.append(-1)

                    ## output_mask用来过滤albert最后输出的特殊符号[CLS]、[SEP]、[PAD]
                    ## 此外，也是为了适应crf
                    output_mask = [1] * len(tokens)
                    #头尾都为零 中间才为有效信息
                    output_mask = [0] + output_mask + [0]
                    while len(output_mask) < max_length:
                        output_mask.append(0)

                    assert len(input_ids) == max_length
                    assert len(input_mask) == max_length
                    assert len(label_ids) == max_length
                    assert len(output_mask) == max_length

                    # ----------------处理后结果-------------------------
                    # for example, in the case of max_seq_length=10:
                    # raw_data:          我 是 中 国 人
                    # token:       [CLS] 我 是 中 国 人  [SEP]
                    # input_ids:     101 2  12 13 16 14   102   0 0 0
                    # input_mask:      1 1  1  1  1  1      1   0 0 0
                    # label_id:          T  T  O  O  O   -1 -1 -1 -1 -1          label_id中不包括[CLS]和[SEP]
                    # output_mask:     0 1  1  1  1  1      0   0 0 0

                    feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids,
                                            output_mask=output_mask)
                    self.train_data.append(feature)


    def prepare_batch(self, train_data, batch_size):
        '''
            prepare data for batch
        '''
        train_ids = torch.LongTensor([temp.input_id for temp in train_data])
        train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
        train_tags = torch.LongTensor([temp.label_id for temp in train_data])
        output_masks = torch.LongTensor([temp.output_mask for temp in train_data])
        train_dataset = TensorDataset(train_ids, train_masks, train_tags, output_masks)

        if self.data_type == 'train':
            return DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        elif self.data_type == 'dev':
            return DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
