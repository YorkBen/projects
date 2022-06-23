import logging
from Lib.DataLoader import DataLoader
from Lib.TextClassifier import TextClassifier

logging.basicConfig(level=logging.DEBUG)

def train_all_features():
    dl = DataLoader()
    ## 训练TextClassifier ############################################
    # 加载所有数据
    lines = dl.load_data_lines(file_path='data/data_20220506_fortrain.txt', num_fields=35, skip_title=False, shuffle=False)
    # 训练循环
    # for k in range(4, 29):
    for k in [4]:
        # 生成训练数据
        feature_name = lines[0][k]
        logging.debug('Training Feature: %s' % feature_name)
        sentences = [l[1] for l in lines[1:]]
        keywords = [feature_name] * len(lines[1:])
        labels = [l[k] for l in lines[1:]]
        dl.stat_cls_num(lines[1:], k)

        # 初始化模型
        model = TextClassifier(model_save_path='output/models/textclassify',
                                pre_model_path="BertModels/medical-roberta-wwm",
                                num_cls=2,
                                model_name=feature_name)

        model.load_data(sentences, labels, texts_pair=keywords, batch_size=8)
        model.train(write_result_to_file='output/textclassify_train_20220518.txt', early_stopping_num=5)

    # 所有数据混合训练
    # print('mix feature training, sample nums: 0:%s, 1:%s, 2:%s' % (labels_cls_count_all[0], labels_cls_count_all[1], labels_cls_count_all[2]))
    # writeToFile(result_file, 'mix feature training, sample nums: 0:%s, 1:%s, 2:%s' % (labels_cls_count_all[0], labels_cls_count_all[1], labels_cls_count_all[2]))
    # datasets = DataToDataset(tokenizer, sentences_all, keywords_all, labels_all)
    # train(datasets, device)

def predict_all_features():
    dl = DataLoader()
    ## 训练TextClassifier ############################################
    # 加载所有数据
    lines = dl.load_data_lines(file_path='data/data_20220506_fortrain.txt', num_fields=35, skip_title=False, shuffle=False)
    model_names = ['排尿改变_20220518111542_8647.pth']
    # 训练循环
    # for k in range(4, 29):
    for k in [6]:
        # 生成训练数据
        feature_name = lines[0][k]
        logging.debug('Predicting Feature: %s' % feature_name)
        sentences = [l[1] for l in lines[1:]]
        keywords = [feature_name] * (len(lines) - 1)
        labels = [l[k] for l in lines[1:]]
        dl.stat_cls_num(lines[1:], k)

        # 初始化模型
        model = TextClassifier(model_save_path='output/models/textclassify',
                                pre_model_path="BertModels/medical-roberta-wwm",
                                num_cls=3,
                                model_name=feature_name,
                                model_file_path='output/models/textclassify/%s' % model_names[k-6]
                                )

        model.load_data(sentences, labels, texts_pair=keywords, batch_size=8, is_training=False)
        model.predict(write_result_to_file='output/results/textclassify_predict_20220518.txt')

if __name__ == "__main__":
    train_all_features()
    # predict_all_features()

    #
    #
