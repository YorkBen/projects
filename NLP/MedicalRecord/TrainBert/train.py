import os
import csv
from transformers import  BertTokenizer, WEIGHTS_NAME,TrainingArguments
# from model.modeling_nezha import NeZhaForSequenceClassification, NeZhaForMaskedLM
# from model.configuration_nezha import NeZhaConfig
import tokenizers
import torch
# from datasets import load_dataset, Dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)

# 自己修改部分配置参数
config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
     # "hidden_size": 128,
    # "num_attention_heads": 4,
    "hidden_dropout_prob": 0.2,
#     "vocab_size": 863 # 自己设置词汇大小
}

## 加载tokenizer和模型
train_file='data/test.txt'
model_path='./model/medical/'
token_path='./model/tk_pretrain_models/vocab.txt'

tokenizer =  BertTokenizer.from_pretrained(token_path, do_lower_case=True)
config = AutoConfig.from_pretrained('hfl/chinese-roberta-wwm-ext-large', **config_kwargs)
# 载入预训练模型
model = AutoModelForMaskedLM.from_pretrained(
            'hfl/chinese-roberta-wwm-ext-large',
            # from_tf=bool(".ckpt" in 'roberta-base'), # 支持tf的权重
            config=config,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
        )
model.resize_token_embeddings(len(tokenizer))
#output:Embedding(863, 768, padding_idx=1)

# ## 制作自己的tokenizer
# bwpt = tokenizers.BertWordPieceTokenizer()
# filepath = "../excel2txt.txt" # 和本文第一部分的语料格式一致
# bwpt.train(
#     files=[filepath],
#     vocab_size=50000,
#     min_frequency=1,
#     limit_alphabet=1000
# )
# bwpt.save_model('./pretrained_models/') # 得到vocab.txt

# 通过LineByLineTextDataset接口 加载数据 #长度设置为128, # 这里file_path于本文第一部分的语料格式一致
train_dataset=LineByLineTextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
# MLM模型的数据DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# 训练参数
pretrain_batch_size=128
num_train_epochs=30
training_args = TrainingArguments(
    output_dir='./outputs/', overwrite_output_dir=True, num_train_epochs=num_train_epochs, learning_rate=6e-5,
    per_device_train_batch_size=pretrain_batch_size,save_total_limit=10)# save_steps=10000
# 通过Trainer接口训练模型
trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

# 开始训练
trainer.train(True)
trainer.save_model('./outputs/')
