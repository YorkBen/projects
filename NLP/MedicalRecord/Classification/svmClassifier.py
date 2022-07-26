import random
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
import logging
from utils import *
from Lib.DataLoader import DataLoader

logging.basicConfig(level=logging.DEBUG)

def load_data(file_path):
    dl = DataLoader()
    lines = dl.load_data_lines(file_path, num_fields=122, skip_title=False, shuffle=True)
    train_lines, val_lines = dl.split_data_by_cls_num(lines, 1)
    train_X = [l[2:] for l in train_lines]
    train_y = [l[1] for l in train_lines]
    val_X = [l[2:] for l in val_lines]
    val_y = [l[1] for l in val_lines]

    return train_X, train_y, val_X, val_y

if __name__ == "__main__":
    train_X, train_y, val_X, val_y = load_data(r'data\train_data_202207.txt')

    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    val_X = ss.transform(val_X)

    # 创建SVM分类器
    model = svm.SVC()
    # 用训练集做训练
    model.fit(train_X, train_y)
    # 用测试集做预测
    prediction=model.predict(val_X)
    print('准确率: {:.2%}'.format(metrics.accuracy_score(val_y,prediction)))
