from paddlenlp import Taskflow
from pprint import pprint

class REProcessor:
    def __init__(self):
        self.schema = {'症状': ['部位', '否定词', '性质']}
        self.mdl_path = r'../../paddlenlp/uie_zh/checkpoint_ext/model_best'
        self.ie = ie = Taskflow("information_extraction", schema=self.schema, task_path=self.mdl_path, device_id=0)
        # ie = Taskflow("information_extraction", schema=schema, task_path='./model_ext', device_id=-1) # CPU

    def infer_per_text(self, text):
        """
        [{'症状': [{'end': 13,
              'probability': 0.9998205973689949,
              'relations': {'性质': [{'end': 18,
                                    'probability': 0.7157156076722941,
                                    'start': 16,
                                    'text': '加重'}]},
              'start': 11,
              'text': '呕吐'},
             {'end': 11,
              'probability': 0.9998343057367265,
              'relations': {'性质': [{'end': 18,
                                    'probability': 0.6993950986271891,
                                    'start': 16,
                                    'text': '加重'}]},
              'start': 9,
              'text': '恶心'}]}]
        """
        results = []
        for ie_item in ie(text):
            if '症状' in ie_item:
                for item in ie_item['症状']:
                    symp_text = item['text']#.replace('疼', '痛')
                    symp_prob = item['probability']

                    # 性质
                    status_texts = []
                    if 'relations' in item and '性质' in item['relations']:
                        status_texts = [r_item['text'] for r_item in item['relations']['性质']]

                    # 否定词
                    neg_texts = []
                    if 'relations' in item and '否定词' in item['relations']:
                        neg_texts = [r_item['text'] for r_item in item['relations']['否定词']]

                    # 部位
                    body_texts = []
                    if 'relations' in item and '部位' in item['relations']:
                        body_texts = [r_item['text'] for r_item in item['relations']['部位']]

                    r = infer_by_feature(symp_text, symp_prob, status_texts, neg_texts, body_texts)
                    results.append(r)

            elif '生理' in ie_item:
                pass
