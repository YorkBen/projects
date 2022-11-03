# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping

def load_data_file(file_path, n_labels, separator='	', skip_head=True):
    """
    加载数据文件，文件格式：
    f1, f2, f3, ... fn, lbl1, lbl2, lbl3...
    """
    features, labels = [], []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and skip_head:
                continue
            arr = line.strip().split(separator)
            features.append([float(e) for e in arr[:-n_labels]])
            labels_arr = [float(1) if float(e) > 1 else float(e) for e in arr[-n_labels:]]
            labels.append(labels_arr)

    return np.array(features), np.array(labels)


class Evaluater():
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.record_num = 0
        self.correct_record_num = 0

    def compute(self, y, preds):
        preds = (preds > 0.5)
        y = y
        for r1, r2 in zip(y, preds):
            self.record_num = self.record_num + 1
            has_diff = False
            for e1, e2 in zip(r1, r2):
                if e1 == 1 and e2 == 1:
                    self.tp = self.tp + 1
                elif e1 == 1 and e2 == 0:
                    self.fn = self.fn + 1
                    has_diff = True
                elif e1 == 0 and e2 == 0:
                    self.tn = self.tn + 1
                else:
                    self.fp = self.fp + 1
                    has_diff = True

            if not has_diff:
                self.correct_record_num = self.correct_record_num + 1


    def calc(self):
        precision = float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0
        recall = float(self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else 0
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        record_acc = float(self.correct_record_num / self.record_num)
        return precision, recall, f1_score, record_acc



# get the dataset
def get_dataset():
	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
	return X, y

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(64, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results

if __name__ == '__main__':
    # load dataset
    # 加载数据
    n_labels, n_hidden = 10, 64
    train_X, train_y = load_data_file(r'data\训练_全特征_多诊断.txt', n_labels)
    val_X, val_y = load_data_file(r'data\测试_全特征_多诊断.txt', n_labels)

    n_inputs, n_outputs = train_X.shape[1], train_y.shape[1]
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    checkpointer = ModelCheckpoint(filepath='/output/models/keras/weights.hdf5', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    model.fit(train_X, train_y, verbose=1, validation_data=(val_X, val_y), epochs=500, callbacks=[checkpointer, early_stop])
    yhat = model.predict(val_X)
    print('Predicted: %s' % yhat[0])

    # evaluater = Evaluater()
    # evaluater.compute(val_y, yhat)
    # precision, recall, f1, record_acc = evaluater.calc()
    # print(f"Precision: { precision*100:>4.2f}%, Recall: {recall *100:>4.2f}%, F1: {f1 *100:>4.2f}%, Record Acc: {record_acc *100:>4.2f}% \n")


    # 模型预测指标
    val_predicts = (np.array(yhat) >= 0.5).astype(int)
    accuracy = accuracy_score(val_y, val_predicts)
    f1_score_micro = f1_score(val_y, val_predicts, average='micro')
    f1_score_macro = f1_score(val_y, val_predicts, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
