import joblib
import pickle

class Diagnoser:
    def __init__(self, scaler_path, mdl_path):
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.model = joblib.load(mdl_path)
        self.dieases = ["急性阑尾炎", "急性胰腺炎", "肠梗阻", "异位妊娠", "急性胆管炎", "急性胆囊炎", "上尿路结石", "卵巢囊肿", "消化道穿孔"]


    def filt_by_num(self, probs, pred_num, out_format):
        """
        对索引按照概率排序，并输出相应疾病名
        """
        prob_idx_data = [(idx, prob) for idx, prob in enumerate(probs)]
        prob_idx_data = sorted(prob_idx_data, key=lambda x: x[1], reverse=True)
        if out_format == 'dict':
            pred_dieases = [{'name': self.dieases[idx], 'confirm': 0} for i, (idx, _) in enumerate(prob_idx_data) if i < pred_num]
        elif out_format == 'id':
            pred_dieases = [idx for i, (idx, _) in enumerate(prob_idx_data) if i < pred_num]
        else:
            pred_dieases = [self.dieases[idx] for i, (idx, _) in enumerate(prob_idx_data) if i < pred_num]

        return pred_dieases

    def filt_by_prob(self, probs, prob_delta, out_format):
        if out_format == 'dict':
            pred_dieases = [{'name': self.dieases[idx], 'confirm': 0, 'prob': prob} for idx, prob in enumerate(probs) if prob >= prob_delta]
        elif out_format == 'id':
            pred_dieases = [idx for idx, prob in enumerate(probs) if prob >= prob_delta]
        else:
            pred_dieases = [self.dieases[idx] for idx, prob in enumerate(probs) if prob >= prob_delta]

        return pred_dieases


    def predict_one(self, X):
        """
        输出最大概率结果
        """
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict(self, X):
        """
        X为一维特征向量
        """
        X = self.scaler.transform([X])
        probs = self.model.predict_proba(X).tolist()[0]

        return self.sort_by_probs(probs)

    def predict_batch(self, X, pred_num=None, prob_delta=None, out_format='dict'):
        """
        X为二维特征向量，有多行
        pred_num: 按照概率输出预测类别数量
        out_format:
            1. dict: 输出字典
            2. name: 并输出相应疾病名
            3. id：  输出疾病id
        """
        X = self.scaler.transform(X)
        probs_list = self.model.predict_proba(X).tolist()

        result = []
        if pred_num is not None and prob_delta is None:
            for probs in probs_list:
                result.append(self.filt_by_num(probs, pred_num, out_format))
        elif prob_delta is not None:
            for probs in probs_list:
                result.append(self.filt_by_prob(probs, prob_delta, out_format))

        return result

#
