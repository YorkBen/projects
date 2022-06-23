import re

def rp(str):
    str = str.split('；')[0]
    str = re.sub('[(（](现病史|首次病程|体格检查|手术史|病史小结|专科情况|既往史|家族史|个人史|年龄|婚育史).*[）)]', '', str)
    str = re.sub('[；，-](现病史|首次病程|体格检查|手术史|病史小结|专科情况|既往史|家族史|个人史|年龄|婚育史)', '', str)
    str = str.replace('（腹痛）', '')
    # str = str.replace('（现病史）', '').replace('（首次病程）', '').replace('（首次病程+专科情况）', '').replace('，现病史', '')
    # str = str.replace('（体格检查和首次病程）', '').replace('（既往史+现病史）', '').replace('（既往史）', '').replace('（既往史+手术史+现病史）', '')
    # str = str.replace('（现病史+手术史+病史小结）', '').replace('（体格检查）', '').replace('（手术史）', '').replace('（现病史/既往史）', '')

    return str

with open('1.txt') as f:
    for l in f.readlines():
        print(rp(l.strip()))
