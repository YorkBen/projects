#!/c/ProgramData/Anaconda3/python
# -*- coding: UTF-8 -*-
import re
import random

# def segmentSentenct(sentence, seg_re=r'[，。；：,?？!！]'):
def segmentSentenct(sentence, seg_re=r'[。；?？!！]'):
    """
    切分句子，并找出有hint_word的短句
    """
    pattern = re.compile(seg_re)
    sentence = re.sub(pattern, '++++', sentence)
    segs = sentence.split('++++')

    results = []
    max_len = 0
    for seg in segs:
        if len(seg) > max_len:
            max_len = len(seg)
        if len(seg) > 0:
            results.append(seg)

    return results

sentences=[]
with open('medical_txt.txt', 'r') as f:
    for line in f.readlines():
        sents = segmentSentenct(line.strip())
        sentences.extend(sents)

random.shuffle(sentences)
total_num = len(sentences)
train_num = int(total_num * 0.8)
print('sentences num: %s' % len(sentences))

# sents = model.sent_split([sentences[0]])
with open('train.txt', 'w') as f:
    for line in sentences[:train_num]:
        f.write(line + '\n')

with open('val.txt', 'w') as f:
    for line in sentences[train_num:]:
        f.write(line + '\n')
