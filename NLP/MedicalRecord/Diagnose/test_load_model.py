import joblib
import pickle
from Lib.DataLoader import DataLoader
from Lib.Diagnoser import Diagnoser

def load_data(file_path, num_fields, separator, skip_title, cls_col, start_feature_col, end_feature_col, shuffle=True):
    dl = DataLoader()
    lines = dl.load_data_lines(file_path, num_fields=num_fields, separator=separator, skip_title=skip_title, shuffle=shuffle)
    X = [[float(e) for e in l[start_feature_col:end_feature_col+1]] for l in lines]
    y = [int(l[cls_col]) for l in lines]
    ids = [l[0:2] for l in lines]

    return X, y, ids

if __name__ == "__main__":
    num_fields = 176

    dg = Diagnoser(r'output/models/diagnose/scaler_gbdt_全特征.pkl', r'output/models/diagnose/gbdt_全特征_0.8308.m')
    # train_X, train_y = load_data(r'data/人机大赛_临床.txt', num_fields=171, separator='	', skip_title=True, cls_col=2, start_feature_col=3, end_feature_col=170)
    X_test, y_test, ids_test = load_data(r'data/人机_全特征.txt', num_fields=num_fields, separator='	',
                            skip_title=True, cls_col=2, start_feature_col=3, end_feature_col=num_fields-1, shuffle=False)

                                # result = model.predict(scaler.transform(train_X[:2000])).tolist()
    # correct = 0
    # for y1, y2 in zip(result, train_y[:2000]):
    #     if y1 == y2:
    #         correct = correct + 1
    # print('accuracy_score: %.4f' % (correct / 2000.0))

    results = dg.predict_one(X_test)
    with open(r'tmp.txt', 'w') as f:
        for r in results:
            f.write('%s\n' % str(r))


#
