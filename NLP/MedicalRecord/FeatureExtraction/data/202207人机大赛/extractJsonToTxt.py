import json

if __name__ == '__main__':
    # postfix = '1611'
    postfix = '1611'

    # 加载json数据
    json_data = ''
    with open(r'汇总结果_%s.json' % postfix) as f:
        json_data = json.load(f, strict=False)

    # 加载labled_ind.txt
    mr_nos = []
    with open(r'labeled_ind_231.txt', 'r') as f:
        for line in f.readlines():
            mr_nos.append(line.strip())

    # 加载txt数据
    data_dict = {}
    for key in ['放射', '超声', '病理']:
        with open(r'%s.txt' % key, 'r') as f:
            for line in f.readlines():
                arr = line.split(',')
                mr_no = arr[0]
                if mr_no not in data_dict:
                    data_dict[mr_no] = {
                        '放射': [],
                        '超声': [],
                        '病理': []
                    }

                data_dict[mr_no][key].append(line.strip())

    # json数据写入
    for item in json_data:
        mr_no = item["医保编号"]
        if mr_no in data_dict:
            for key in ['放射', '超声', '病理']:
                arr = []
                if key in item and len(item[key]) > 0:
                    for item_key in item[key]:
                        t_date = item_key['日期']
                        for item_key_sj in item_key['数据']:
                            arr.append('%s,%s,%s' % (mr_no, t_date, item_key_sj))
                    if len(arr) >= len(data_dict[mr_no][key]):
                        data_dict[mr_no][key] = arr


    # 从新写到txt
    for key in ['放射', '超声', '病理']:
        with open(r'%s_r.txt' % key, "w") as f:
            for mr_no in data_dict.keys():
                for line in data_dict[mr_no][key]:
                    f.write(line + '\n')


#
