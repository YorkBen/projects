
data_path = r"data/medical_record.csv"

"""
入院：
    泌尿外科入院记录
    非手术科室入院记录
    手术科室入院记录
    --产科入院记录
    神经内科入院记录
    肿瘤科入院记录
    耳鼻喉科入院记录
    24小时入院死亡记录
    眼科入院记录
    介入放射学科入院记录
    临床心理科入院记录
    儿科入院记录
    新生儿入院记录
    --康复科入院记录
    精神科入院记录
    神经外科入院记录
    儿外科入院记录
    肿瘤科再次入院记录
"""

stat_dict = {
    "术后首次病程记录": 0,
    "首次病程": 0,
    "日常病程记录": 0,
    "非手术科室入院记录": 0,
    "手术科室入院记录": 0,
    "同意书": 0,
    "手术记录": 0,
    "出院记录": 0,
    "出院诊断证明书": 0,
    "泌尿外科入院记录": 0,
    "儿科入院记录": 0,
    "肿瘤科入院记录": 0,
    "神经内科入院记录": 0,
    "神经外科入院记录": 0
}
def stat_dict_ct(str):
    for k in stat_dict.keys():
        if k in str:
            stat_dict[k] = stat_dict[k] + 1
            return

append_dict = {
    "术后首次病程记录": [],
    "首次病程": [],
    "日常病程记录": [],
    "非手术科室入院记录": [],
    "手术科室入院记录": [],
    "同意书": [],
    "手术记录": [],
    "出院记录": [],
    "出院诊断证明书": [],
    "泌尿外科入院记录": [],
    "儿科入院记录": [],
    "肿瘤科入院记录": [],
    "神经内科入院记录": [],
    "神经外科入院记录": []
}
def append_to_dict(item):
    for k in append_dict.keys():
        if k in item[3]:
            append_dict[k].append(item)
            return

def record_split(record_type=None):
    """
    切分医疗记录，同一个入院记录的文本行合并，只输出入院记录
    """
    medical_records, mr_item, mr_cnt = [], [], ''

    with open(data_path) as f:
        for idx, line in enumerate(f.readlines()):
            # line = line.strip()
            if idx > 0:
                # 还没有进入病历内部
                if len(mr_item) == 0:
                    elems = line.split(',')
                    # 找到病历第一行
                    if len(elems) == 5 and '\"' in elems[4]:
                        mr_item = elems[0:4]
                        mr_cnt = elems[4].replace('\"', '')
                        stat_dict_ct(mr_item[3])
                elif line.strip() != '':
                    # 到达一个病历的结束行
                    # 保留\n在后面作为行分割使用
                    if line.strip().endswith('\"'):
                        # 只记录入院记录
                        if record_type and record_type in mr_item[3] or record_type is None:
                            mr_item.append(mr_cnt + line.strip()[:-1] + "\n")
                            # medical_records.append(mr_item)
                            append_to_dict(mr_item)
                        # 清空数据，初始化
                        mr_item = []
                        mr_cnt = ''
                    else:
                        mr_cnt = mr_cnt + line

            # if len(medical_records) == 2:
            #     break

#     return list(set(record_types))
#
# with open("data/tmp/types.txt", "w") as f:
#     for line in record_split():
#         f.write(line.strip() + '\n')
record_split()
print(stat_dict)
for k in append_dict.keys():
    with open("data/records/%s.txt" % k, "w") as f:
        for item in append_dict[k]:
            f.write(item[0] + "||" + item[1] + "||" + item[2] + "||" + item[3] + "\n")
            f.write(item[4])
            f.write("   \n")
