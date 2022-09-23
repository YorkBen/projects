import os
import random
import logging
from openpyxl import load_workbook, Workbook

import sys

sys.path.append('../../Lib/Models')

from DataLoader import DataLoader
# from TextClassifier import TextClassifier
from REClassifier import TextClassifier

logging.basicConfig(level=logging.DEBUG)

label_cls_dict = {
    'null': 0,
    'target': 1,
    'Gene function': 2,
    'dependence': 3,
    'responsive': 4,
    'pathway': 5,
    'negtive': 6,
    'Gene multifunction': 7,
    'promote dependence': 8,
    'positive': 9,
    'inhibite dependence': 10,
    'inhibite target': 11,
    'promote pathway': 12,
    'relatied': 13,
    'inhibite pathway': 14,
    'transcriptional coactivation': 15
}

label_cls_dict = {
    'null': 0,
    'negtive': 1,
    'positive': 2,
    'relatied': 3
}

def load_data(file_path):
    """
    输入数据格式：text \t relation name
    """
    data = []
    with open(file_path) as f:
        for line in f.readlines():
            arr = line.strip().split('\t')
            data.append((arr[0], arr[1]))

    return data

def load_train_val_data(file_path):
    data = load_data(file_path)

    random.shuffle(data)
    train_len = int(len(data) * 0.8)

    train_data, val_data = data[:train_len], data[train_len:]
    return train_data, val_data

def load_test_data(file_path):
    data = load_data(file_path)
    texts, labels = [e[0] for e in data], [e[1] for e in data]

    return texts, labels

def cust_logging(log_file, str):
    with open(log_file, 'a+') as f:
        f.write(str + '\n')


def stat_acc_by_cls(labels, preds):
    """
    统计预测结果标签数量及正确率
    """
    result = {}
    for i1, i2 in zip(labels, preds):
        i1, i2 = int(i1), int(i2)
        if i1 not in result:
            result[i1] = {
                'count': 0,
                'correct': 0
            }
        result[i1]['count'] = result[i1]['count'] + 1
        if i1 == i2:
            result[i1]['correct'] = result[i1]['correct'] + 1

    for i in result.keys():
        result[i]['ratio'] = result[i]['correct'] / result[i]['count']

    return result

def write_stat_result(feature_name, result, log_file):
    result_arr = [(0, 0) for i in range(4)]
    total_count, total_correct = 0, 0
    for key, val in result.items():
        result_arr[key] = (val['count'], val['ratio'])
        total_count = total_count + val['count']
        total_correct = total_correct + val['correct']
    result_arr.append((total_count, total_correct/total_count))

    with open(log_file, 'a+') as f:
        s = feature_name
        for ct, ratio in result_arr:
            s = s + ('	%d	%.2f' % (ct, ratio))
        f.write('%s\n' % s)


def train(model_save_path='output/models', log_file='output/logs/log.txt'):
    # train_data, val_data = load_train_val_data('test_gene_re.txt')
    train_data, val_data = load_train_val_data('train_gene_re_4c.txt')

    # 初始化模型
    model = TextClassifier(model_save_path=model_save_path,
                            pre_model_path="bert-base-cased",
                            # pre_model_path="../../Models/BertModels/gene",
                            # pre_model_path="hfl/chinese-roberta-wwm-ext",
                            num_cls=len(label_cls_dict),
                            model_name='gene_re_4c')

    model.load_train_val_data(train_data, val_data, label_dict=label_cls_dict, batch_size=8)
    return model.train(write_result_to_file=log_file, early_stopping_num=5)


def predict():
    model_save_path='output/models'
    log_file='output/logs/log.txt'
    sentences = [
        'Clear-cell papillary renal cell carcinoma: molecular and immunohistochemical analysis with emphasis on the von Hippel-Lindau gene and hypoxia-inducible factor pathway-related proteins.',
        'Immunohistochemical stains for markers of HIF pathway activation (HIF-1α, CA9, and glucose transporter-1 (GLUT-1)) as well as other relevant markers (CK7, CD10, AMACR, and TFE3) were performed.',
        'The co-expression of CA9, HIF-1α, and GLUT-1 in the absence of VHL gene alterations in clear-cell papillary renal cell carcinoma suggests activation of the HIF pathway by non-VHL-dependent mechanisms.',
        'Carbonic anhydrase 9 (CA9) and vimentin are hypoxia and epithelial-mesenchymal transition-related proteins of which expression in many carcinomas has been associated with poor prognosis, but their significance in PanNET has yet to be determined.',
        'Evaluation of CAIX and CAXII Expression in Breast Cancer at Varied O2 Levels: CAIX is the Superior Surrogate Imaging Biomarker of Tumor Hypoxia.',
        'Expression of cell-surface carbonic anhydrases IX and XII (CAIX and CAXII) in tumor cells has been associated with tumor hypoxia.',
        'Also, the main effectors of canonical (β-catenin), planar cell polarity (JNK), and calcium dependent (NFAT5) Wnt pathways were evaluated by immunohistochemistry.',
        'Some clinical investigations have suggested that high expression of hypoxia-inducible factor-1α (HIF-1α) and/or its target gene carbonic anhydrase IX (CAIX) may be useful biomarkers of tumor hypoxia and poor outcome in cervical cancer.',

    ]
    labels = ['null', 'pathway', 'positive', 'Gene function', 'Gene function', 'Gene function', 'pathway', 'Gene function']

    model = TextClassifier(model_save_path='output/models',
                            pre_model_path="bert-base-cased",
                            num_cls=len(label_cls_dict),
                            model_name='gene_re',
                            model_file_path=os.path.join(model_save_path, 'gene_re_20220920110229_8264.pth'),
                            device='cpu'
                            )

    model.load_data(sentences, labels, label_dict=label_cls_dict, batch_size=4, is_training=False)
    results = model.predict_nowrite()
    print(results)
    # with open(result_file_path, 'w') as f:
    #     for (mrno, rdate), (lbl, pred) in zip(ids, results):
    #         f.write('%s	%s	%s	%s\n' % (mrno, rdate, lbl, pred))

    # stat_result = stat_acc_by_cls([e[0] for e in results], [e[1] for e in results])
    # write_stat_result(feature_name, stat_result, log_file_path)


def predict_file(file_path, model_name, device='cpu'):
    sentences, labels = load_test_data(file_path)
    model_save_path='output/models'
    log_file='output/logs/log.txt'

    model = TextClassifier(model_save_path='output/models',
                            pre_model_path="bert-base-cased",
                            num_cls=len(label_cls_dict),
                            model_name='gene_re',
                            model_file_path=os.path.join(model_save_path, model_name),
                            device=device
                            )

    model.load_data(sentences, labels, label_dict=label_cls_dict, batch_size=8, is_training=False)
    results = model.predict_nowrite(output_label=False)
    return results


if __name__ == "__main__":
    train()
    # train()
    #
    #
