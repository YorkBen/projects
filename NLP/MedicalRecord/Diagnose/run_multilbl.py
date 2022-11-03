from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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
                label_names = arr[-n_labels:]
            else:
                lines.append(arr)

#     random.shuffle(lines)

    X = np.array([[float(e) for e in arr[:-n_labels]] for arr in lines])
    #labels = [[int(1) if float(e) > 1 else float(e) for e in arr[-n_labels:]] for arr in lines]
    y = np.array([[int(e) for e in arr[-n_labels:]] for arr in lines])
    # y = np.array([[int(arr[-n_labels+idx]) for arr in lines] for idx in range(n_labels)])

    return X, y, feature_names, label_names


n_labels = 10
# 加载数据
X_train, y_train, feature_names, label_names = load_data_file(r'data/训练_全特征_多诊断.txt', n_labels)
n_features = len(X_train[0])
X_test, y_test, feature_names, label_names = load_data_file(r'data/测试_全特征_多诊断.txt', n_labels)



# train_size = int(len(labels) * 0.8)
# val_size = len(labels) - train_size
# print('train data size: %s, val data size: %s' % (train_size, val_size))
# print('train feature num: %s, predict label num: %s' % (len(features[0]), len(labels[0])))
# X_train, y_train = np.array(features[:train_size]), np.array(labels[:train_size])
# X_test, y_test = np.array(features[train_size:]), np.array(labels[train_size:])

def self_eval(y, preds):
    """
    自定义评价多标签分类
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    for r1, r2 in zip(y, preds):
        for e1, e2 in zip(r1, r2):
            if e1 == 1 and e2 == 1:
                TP = TP + 1
            elif e1 == 1 and e2 == 0:
                FN = FN + 1
            elif e1 == 0 and e2 == 0:
                TN = TN + 1
            else:
                FP = FP + 1


    precision = float(TP / (TP + FP))
    recall = float(TP / (TP + FN))
    f1_score = float(2 * precision * recall / (precision + recall))
    return precision, recall, f1_score


### One VS. Rest
def one_vs_rest():
    from sklearn.pipeline import Pipeline
    from sklearn.multiclass import OneVsRestClassifier
    # Using pipeline for applying logistic regression and one vs rest classifier
    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
                ])

    preds = []
    for idx, category in enumerate(label_names):
        print('**Processing {} comments...**'.format(category))

        # Training logistic regression model on train data
        LogReg_pipeline.fit(X_train, [e[idx] for e in y_train])

        # calculating test accuracy
        prediction = LogReg_pipeline.predict(X_test)
        preds.append(prediction)

    predictions = [[p[idx] for p in preds] for idx in range(len(preds[0]))]

    print('Test accuracy is {}'.format(accuracy_score(y_test, predictions)))
    print("\n")

    p, r, f1 = self_eval(y_test, predictions)
    print('Custom evaluation: Precision：%s, Recall：%s, F1: %s' % (p, r, f1))

# Test accuracy is 0.4975288303130148
# Custom evaluation: Precision：0.700589970501475, Recall：0.6709039548022598, F1: 0.6854256854256854

### Binary Relevance
def binary_relv():
    # using binary relevance
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    # accuracy
    print("Accuracy = ", accuracy_score(y_test, predictions))
    # accuracy: 0.04


### Classifier Chains
# using classifier chains
def classifier_chains():
    from skmultilearn.problem_transform import ClassifierChain
    # initialize classifier chains multi-label classifier
    classifier = ClassifierChain(LogisticRegression())
    # Training logistic regression model on train data
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    # accuracy
    print("Accuracy = ",accuracy_score(y_test, predictions))
    print("\n")

    predictions = predictions.toarray()
    p, r, f1 = self_eval(y_test, predictions)
    print('Custom evaluation: Precision：%s, Recall：%s, F1: %s' % (p, r, f1))

# Accuracy =  0.5354200988467874
# Custom evaluation: Precision：0.6857887874837028, Recall：0.7429378531073446, F1: 0.7132203389830507


### Label Powerset
def label_powerset():
    # using Label Powerset
    from skmultilearn.problem_transform import LabelPowerset
    # initialize label powerset multi-label classifier
    classifier = LabelPowerset(LogisticRegression())
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    # accuracy
    print("Accuracy = ", accuracy_score(y_test, predictions))
    print("\n")

    predictions = predictions.toarray()
    p, r, f1 = self_eval(y_test, predictions)
    print('Custom evaluation: Precision：%s, Recall：%s, F1: %s' % (p, r, f1))

# Accuracy =  0.5485996705107083
# Custom evaluation: Precision：0.6458885941644562, Recall：0.6878531073446328, F1: 0.6662106703146375


### Adapted Algorithm
def adapted_algorithm():
    from skmultilearn.adapt import MLkNN
    classifier_new = MLkNN(k=10)
    # Note that this classifier can throw up errors when handling sparse matrices.
    X_train = lil_matrix(X_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    X_test = lil_matrix(X_test).toarray()
    # train
    classifier_new.fit(X_train, y_train)
    # predict
    predictions_new = classifier_new.predict(X_test)
    # accuracy
    print("Accuracy = ",accuracy_score(y_test, predictions_new))
    print("\n")

# Accuracy =  0.3542009884678748


### Multi-label embeddings
def multi_label_embedding():
    from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
    from sklearn.manifold import SpectralEmbedding
    from sklearn.ensemble import RandomForestRegressor
    from skmultilearn.adapt import MLkNN

    clf = EmbeddingClassifier(
        SKLearnEmbedder(SpectralEmbedding(n_components = 10)),
        RandomForestRegressor(n_estimators=10),
        MLkNN(k=5)
    )

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    # accuracy
    print("Accuracy = ",accuracy_score(y_test, predictions))
    print("\n")



### Ensembles of classifiers
def ensemble():
    from skmultilearn.ensemble import MajorityVotingClassifier
    from skmultilearn.cluster import FixedLabelSpaceClusterer
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.naive_bayes import GaussianNB

    classifier = MajorityVotingClassifier(
        clusterer = FixedLabelSpaceClusterer(clusters = [[1,3,4], [0, 2, 5]]),
        classifier = ClassifierChain(classifier=GaussianNB())
    )
    classifier.fit(X_train,y_train)
    predictions = classifier.predict(X_test)

    print("Accuracy = ",accuracy_score(y_test, predictions))
    print("\n")

# Accuracy =  0.11037891268533773
