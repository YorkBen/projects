import tokenizers
# 创建分词器
bwpt = tokenizers.BertWordPieceTokenizer()
filepath = "data/test.txt" # 语料文件
#训练分词器
bwpt.train(
    files=[filepath],
    vocab_size=50000, # 这里预设定的词语大小不是很重要
    min_frequency=1,
    limit_alphabet=1000
)
# 保存训练后的模型词表
bwpt.save_model('model/tk_pretrain_models/')
#output： ['./pretrained_models/vocab.txt']

# 加载刚刚训练的tokenizer
# tokenizer=BertTokenizer(vocab_file='./tk_pretrain_models/vocab.txt')
