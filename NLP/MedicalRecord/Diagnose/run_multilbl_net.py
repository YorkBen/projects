import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import sklearn.metrics as metrics
import numpy as np
from skmultilearn.problem_transform import LabelPowerset


def load_data_file(file_path, n_labels, separator='	', skip_head=True):
    """
    加载数据文件，文件格式：
    f1, f2, f3, ... fn, lbl1, lbl2, lbl3...
    """
    X, y, feature_names, label_names = [], [], [], []
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            arr = line.strip().split(separator)
            if idx == 0:
                feature_names = arr[:-n_labels]
                label_names = arr[-n_labels:-1]
            else:
                lines.append(arr)

#     random.shuffle(lines)

    X = np.array([[float(e) for e in arr[:-n_labels]] for arr in lines])
    #labels = [[int(1) if float(e) > 1 else float(e) for e in arr[-n_labels:]] for arr in lines]
    y = np.array([[int(e) for e in arr[-n_labels:]] for arr in lines], dtype=np.int64).astype(np.int64)

    return X, y, feature_names, label_names



# 加载数据
n_labels = 11
X_train, y_train, feature_names, label_names = load_data_file(r'data/训练_全特征_多诊断.txt', n_labels)
n_features = len(X_train[0])
X_test, y_test, _, _ = load_data_file(r'data/测试_全特征_多诊断.txt', n_labels)


nodes = 8
input_dim = n_features
hidden_dim = int(input_dim/nodes)
output_dim = len(np.unique(y_train))
output_dim = 55

class MultiClassClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(MultiClassClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


net = NeuralNetClassifier(
    MultiClassClassifierModule,
    max_epochs=20,
    verbose=0
)

y_tensor = torch.tensor(y_train, dtype=torch.long)
X_tensor = torch.tensor(X_train, dtype=torch.float32)
print(y_tensor.dtype)

clf = LabelPowerset(classifier=net, require_dense=[True,True])
clf.fit(X_tensor, y_tensor)
y_pred = clf.predict(X_test.astype(np.float32))


score = metrics.accuracy_score(y_test, y_pred)

print(score)
