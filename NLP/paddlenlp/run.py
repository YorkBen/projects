import paddlenlp

# MODEL_NAME = "ernie-3.0-medium-zh"
# MODEL_NAME = "ernie-2.0-large-en"
# ernie_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

from pprint import pprint
from paddlenlp import Taskflow

schema = ['Gene', 'Cancer']
ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
pprint(ie_en('Novel predictive biomarkers for cervical cancer prognosis.'))
pprint(ie_en('In conclusion, this suggests that CA III promotes EMT and cell migration and is potentially related to the FAK/Src signaling pathway in oral cancer.'))
