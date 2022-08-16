import random
from skmultilearn.adapt import MLkNN
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import accuracy_score

def load_data_file(file_path, n_labels, separator='	', skip_head=True):
    """
    加载数据文件，文件格式：
    f1, f2, f3, ... fn, lbl1, lbl2, lbl3...
    """
    features, labels = [], []
    lines = []
    with open(file_path) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and skip_head:
                continue
            arr = line.strip().split(separator)
            lines.append(arr)

    random.shuffle(lines)
    features = [[float(e) for e in arr[:-n_labels]] for arr in lines]
    labels = [[float(1) if float(e) > 1 else float(e) for e in arr[-n_labels:]] for arr in lines]

    return features, labels


if __name__ == '__main__':
    n_labels = 11
    # 加载数据
    features, labels = load_data_file(r'data\multilbl_data_20220721.txt', n_labels)
    n_features = len(features[0])

    train_size = int(len(labels) * 0.8)
    val_size = len(labels) - train_size
    print('train data size: %s, val data size: %s' % (train_size, val_size))
    print('train feature num: %s, predict label num: %s' % (len(features[0]), len(labels[0])))

    X_train, y_train = np.array(features[:train_size]), np.array(labels[:train_size])
    X_test, y_test = np.array(features[train_size:]), np.array(labels[train_size:])
    # X_train = csr_matrix(X_train)
    # y_train = csr_matrix(y_train)
    # X_train = lil_matrix(X_train)
    # y_train = lil_matrix(y_train)

    classifier = MLkNN(k=n_labels)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(accuracy_score(y_test, predictions))
