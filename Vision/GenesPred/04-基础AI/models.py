from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn import tree
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV  


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import pandas as pd

# 贝叶斯
def naive_bayes_classifier(train_x, train_y):
    
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model

def gaussian_classifier(train_x, train_y):
    model = GaussianNB()  
    model.fit(train_x, train_y)  
    return model

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    
    model = LogisticRegression(penalty='l2',solver='lbfgs', max_iter=100)  
    model.fit(train_x, train_y)  
    return model  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in list(best_parameters.items()):  
        print(para, val)  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  


def save_csv(columns, data, csv_path, index=False, header=True):
    """
    :param header:
    :param columns:
    :param index:
    :param data: list of data
    :param csv_path:
    :return:
    """
    data_array = np.array(data)
    df = pd.DataFrame(data_array.T, columns=columns)
    df.to_csv(csv_path, encoding='gbk', index=index, header=header)