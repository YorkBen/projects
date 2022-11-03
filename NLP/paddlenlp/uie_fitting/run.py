# from paddlenlp import Taskflow
# # re = Taskflow("ner", model="ernie-health-chinese")
# # information_extraction
# re = Taskflow("information_extraction", model="ernie-health-chinese")
#
# text = "最近半月以来患者无明显诱因间断出现恶心，伴呕吐，为胃内容物，近3天患者感呕吐明显加重"
#
# print(re(text))
# # electra = AutoModel.from_pretrained('chinese-electra-small')


from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-health-chinese')
tokenizer = AutoTokenizer.from_pretrained('ernie-health-chinese')

text = "最近半月以来患者无明显诱因间断出现恶心，伴呕吐，为胃内容物，近3天患者感呕吐明显加重"
text = "今日查房，孕妇未诉腹痛，无阴道出血、排液，恶心、呕吐较前稍好转，早上可进食。"

text = tokenizer(text)
