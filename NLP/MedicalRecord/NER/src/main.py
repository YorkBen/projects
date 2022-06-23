# -*- coding:utf-8 -*-

import yaml
import sys
import torch
import torch.optim as optim
from lstm_crf.data_format import DataFormat
from lstm_crf.model import BiLSTMCRF
from lstm_crf.utils import f1_score, get_tags, format_result
from transformers import AlbertConfig, BertTokenizer, AlbertModel
# from transformers import BertTokenizer, BertModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NER(object):

    def __init__(self, exec_type="train"):
        self.load_config()
        self.__init_model(exec_type)

    def __init_model(self, exec_type):
        if exec_type == "train":
            self.train_data = DataFormat(batch_size=self.batch_size, max_length=self.max_legnth, data_type='train')
            self.dev_data = DataFormat(batch_size=self.batch_size, max_length=self.max_legnth, data_type="dev")

            self.model = BiLSTMCRF(
                tag_map=self.train_data.tag_map,
                batch_size=self.batch_size,
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            # self.restore_model()

        elif exec_type == "predict":
            self.model = BiLSTMCRF(
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            # self.restore_model()

    def load_config(self):
        with open(r"lstm_crf/models/config.yml") as f:
            config = yaml.safe_load(f)

            self.embedding_size = config.get("embedding_size")
            self.hidden_size = config.get("hidden_size")
            self.batch_size = config.get("batch_size")
            self.max_legnth = config.get('max_length')
            self.model_path = config.get("model_path")
            self.tags = config.get("tags")
            self.dropout = config.get("dropout")
            self.epochs = config.get("epochs")

    def restore_model(self):

        self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
        print("self.model:{}".format(self.model))
        print("model restore success!")


    def train(self):
        self.model.to(DEVICE)
        #weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
        optimizer = optim.Adam(self.model.parameters(), lr = 0.001, weight_decay=0.0005)
        '''
        当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能:
        optimer指的是网络的优化器
        mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
        factor 学习率每次降低多少，new_lr = old_lr * factor
        patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
        verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
        threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
        cooldown(int)： 冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
        min_lr(float or list):学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
        eps(float):学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
        '''
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',factor=0.1,patience=100,verbose=False)
        total_size = self.train_data.train_dataloader.__len__()
        for epoch in range(self.epochs):
            index = 0
            for batch in self.train_data.train_dataloader:
                self.model.train()
                index += 1
                self.model.zero_grad()  # 与optimizer.zero_grad()作用一样
                batch = tuple(t.to(DEVICE) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_out_masks = batch

                bert_encode = self.model(b_input_ids, b_input_mask)
                loss = self.model.loss_fn(bert_encode=bert_encode, tags=b_labels, output_mask=b_out_masks)
                progress = ("█" * int(index * 25 / total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                    epoch, progress, index, total_size, loss.item()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1) #梯度裁剪
                optimizer.step()
                schedule.step(loss)
            self.eval_2()
            print("-" * 50)
        torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

    def eval_2(self):
        '''
        只评估PER,ORG,LOC,T
        :return:
        '''
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.dev_data.train_dataloader):
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, input_mask, label_ids, output_mask = batch
                bert_encode = self.model(input_ids, input_mask)
                predicts = self.model.predict(bert_encode, output_mask)
                print("\teval")
                for tag in self.tags:
                    f1_score(label_ids, predicts, tag, self.model.tag_map)

    '''
    注意：
        1.在模型中有BN层或者dropout层时，在训练阶段和测试阶段必须显式指定train()
            和eval()。
        2.一般来说，在验证或者是测试阶段，因为只是需要跑个前向传播(forward)就足够了，
            因此不需要保存变量的梯度。保存梯度是需要额外显存或者内存进行保存的，占用了空间，
            有时候还会在验证阶段导致OOM(Out Of Memory)错误，因此我们在验证和测试阶段，最好显式地取消掉模型变量的梯度。
            使用torch.no_grad()这个上下文管理器就可以了。
    '''

    def predict(self, input_str=""):
        self.model.eval()  # 取消batchnorm和dropout,用于评估阶段
        self.model.to(DEVICE)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # tokenizer = BertTokenizer.from_pretrained('BertModels/medical-roberta-wwm')

        while True:
            with torch.no_grad():
                input_str = input("请输入文本: ")
                input_ids = tokenizer.encode(input_str,add_special_tokens=True)  # add_spicial_tokens=True，为自动为sentence加上[CLS]和[SEP]
                input_mask = [1] * len(input_ids)
                output_mask = [0] + [1] * (len(input_ids) - 2) + [0]  # 用于屏蔽特殊token

                input_ids_tensor = torch.LongTensor(input_ids).reshape(1, -1)
                input_mask_tensor = torch.LongTensor(input_mask).reshape(1, -1)
                output_mask_tensor = torch.LongTensor(output_mask).reshape(1, -1)
                input_ids_tensor = input_ids_tensor.to(DEVICE)
                input_mask_tensor = input_mask_tensor.to(DEVICE)
                output_mask_tensor = output_mask_tensor.to(DEVICE)

                bert_encode = self.model(input_ids_tensor, input_mask_tensor)
                predicts = self.model.predict(bert_encode, output_mask_tensor)

                print('paths:{}'.format(predicts))
                entities = []
                for tag in self.tags:
                    tags = get_tags(predicts[0], tag, self.model.tag_map)
                    entities += format_result(tags, input_str, tag)
                print(entities)

if __name__ == "__main__":
    # ner = NER("train")
    # ner.train()
    ner = NER("predict")
    print(ner.predict())
