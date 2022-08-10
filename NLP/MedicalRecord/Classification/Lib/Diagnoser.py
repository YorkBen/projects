import joblib
import pickle

class Diagnoser:
    def __init__(self, scaler_path, mdl_path):
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.model = joblib.load(mdl_path)
        self.dieases = ["急性阑尾炎", "急性胰腺炎", "肠梗阻", "异位妊娠", "急性胆管炎", "急性胆囊炎", "上尿路结石", "卵巢囊肿破裂", "卵巢囊肿扭转", "消化道穿孔"]


    def sort_by_probs(self, probs):
        """
        对索引按照概率排序，并输出相应疾病名
        """
        prob_idx_data = [(idx, prob) for idx, prob in enumerate(probs)]
        prob_idx_data = sorted(prob_idx_data, key=lambda x: x[1], reverse=True)
        pred_dieases = [{'name': self.dieases[idx], 'confirm': 0} for idx, _ in prob_idx_data]

        return pred_dieases

    def predict(self, X):
        """
        X为一维特征向量
        """
        X = self.scaler.transform([X])
        probs = self.model.predict_proba(X).tolist()[0]

        return self.sort_by_probs(probs)

    def predict_batch(self, X):
        """
        X为二维特征向量，有多行
        """
        X = self.scaler.transform(X)
        probs_list = self.model.predict_proba(X).tolist()

        result = []
        for probs in probs_list:
            result.append(self.sort_by_probs(probs))

        return result

#
