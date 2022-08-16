import random
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import logging
from utils import *
from Lib.DataLoader import DataLoader

logging.basicConfig(level=logging.DEBUG)

def load_data(file_path, num_fields, separator, skip_title, cls_col, start_feature_col, end_feature_col):
    dl = DataLoader()
    lines = dl.load_data_lines(file_path, num_fields=num_fields, separator=separator, skip_title=skip_title, shuffle=True)
    train_lines, val_lines = dl.split_data_by_cls_num(lines, cls_col)
    train_X = [[int(e) for e in l[start_feature_col:end_feature_col+1]] for l in train_lines]
    train_y = [int(l[cls_col]) for l in train_lines]
    val_X = [[int(e) for e in l[start_feature_col:end_feature_col+1]] for l in val_lines]
    val_y = [int(l[cls_col])for l in val_lines]
    val_id = [l[0:2] for l in val_lines]

    return train_X, train_y, val_X, val_y, val_id

def write_pred_result(val_id, val_y, prediction):
    with open(r'data/svm_r1.txt', 'w') as f:
        for (id, dt), y, y_ in zip(val_id, val_y, prediction):
            f.write('%s,%s,%s,%s\n' % (id, dt, y, y_))

if __name__ == "__main__":
    # # 全量数据，准确率：68% ~ 72%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合数据_20220802.txt', num_fields=171, separator='	',
    #     skip_title=True, cls_col=170, start_feature_col=2, end_feature_col=169)

    # 去掉实验室数据，准确率：64% ~ 68%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合数据_20220802.txt', num_fields=171, separator='	',
    #     skip_title=True, cls_col=170, start_feature_col=2, end_feature_col=151)

    # # 去掉实验室数据 + 影像学，准确率：34% ~ 36%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合数据_20220802.txt', num_fields=171, separator='	',
    #     skip_title=True, cls_col=170, start_feature_col=2, end_feature_col=119)

    # # 实验室数据 + 影像学，准确率：70% ~ 74%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合数据_20220802.txt', num_fields=171, separator='	',
    #     skip_title=True, cls_col=170, start_feature_col=120, end_feature_col=169)

    # # 临床特征，去掉模型特征，准确率：33.33% ~ 35%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合_临床无模型.txt', num_fields=100, separator='	',
    #     skip_title=True, cls_col=99, start_feature_col=2, end_feature_col=98)

    # # # 临床特征，只用掉模型特征，准确率：34% ~ 37%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合_临床模型.txt', num_fields=24, separator='	',
    #     skip_title=True, cls_col=23, start_feature_col=2, end_feature_col=22)

    # # # 实验室特征，准确率：54% ~ 57%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合_实验室.txt', num_fields=21, separator='	',
    #     skip_title=True, cls_col=2, start_feature_col=3, end_feature_col=20)

    # # 影像学特征，准确率：62% ~ 67%
    # train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合_影像学.txt', num_fields=35, separator='	',
    #     skip_title=True, cls_col=2, start_feature_col=3, end_feature_col=34)

    # # 临床特征，非模型，无人工，准确率：32% ~ 37%
    train_X, train_y, val_X, val_y, val_id = load_data(r'data/疾病诊断拟合_临床_非模型_无人工.txt', num_fields=93, separator='	',
        skip_title=True, cls_col=2, start_feature_col=3, end_feature_col=92)


    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    val_X = ss.transform(val_X)

    # 创建GBDT分类器
    model = GradientBoostingClassifier(max_features=90, max_depth=40, min_samples_split=8, min_samples_leaf=3,
                                n_estimators=1200, learning_rate=0.05, subsample=0.95)
    # 用训练集做训练
    model.fit(train_X, train_y)
    # 用测试集做预测
    prediction=model.predict(val_X)
    print('准确率: {:.2%}'.format(metrics.accuracy_score(val_y,prediction)))

    # write_pred_result(val_id, val_y, prediction)
