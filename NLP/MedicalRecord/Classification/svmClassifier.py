import random
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from utils import *


if __name__ == "__main__":
    idc11_id, num_cls = "DB10", 2

    sentences, features, labels, feature_size = load_data(id_map = id_map_6, input_file=r'data/train_data_20220424.txt')
    # sentences, features, labels = load_data_one_id(id = idc11_id)

    train_num = int(len(features) * 0.8)

    train_X = features[:train_num]
    train_y = labels[:train_num]
    test_X = features[train_num:]
    test_y = labels[train_num:]

    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.transform(test_X)

    # 创建SVM分类器
    model = svm.SVC()
    # 用训练集做训练
    model.fit(train_X, train_y)
    # 用测试集做预测
    prediction=model.predict(test_X)
    print('准确率: {:.2%}'.format(metrics.accuracy_score(test_y,prediction)))
