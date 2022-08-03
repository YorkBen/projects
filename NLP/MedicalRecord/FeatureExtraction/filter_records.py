import re
import json

def load_keys(file_path, with_head=True, separator='	'):
    """
    提取mrnos和入院日期，文件的第一个字段是mrnos，第二个字段是入院时间
    """
    results = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head or line.strip() == '':
                continue

            arr = line.strip().split(separator)
            mr_no, ry_date = arr[0], ''
            if len(arr) == 2:
                ry_date = arr[1]
                if ry_date != '' and not re.match('[0-9]{8}', ry_date):
                    print('%s:%s line illegal!' % (idx, line))
                    exit()

            results.append((mr_no, ry_date))

    return results


def filter_records(json_data, keys):
    """
    根据医保编号和入院日期过滤json数据
    keys：set((mrno, 入院日期)) 或者 set(mrno) 或者None
    """
    keys = set(list(keys))
    results = []
    for record in json_data:
        if record['入院记录'] == '':
            continue
        key = '%s_%s' % (record['医保编号'], record['入院记录']['入院时间'])
        if key in keys:
            # 既往史
            jws = record['入院记录']['既往史']
            for k1 in ['预防接种史', '预防接种药品']:
                jws[k1] = '无' if jws[k1] == '' else jws[k1]
            jbs = record['入院记录']['既往史']['疾病史']
            for k1 in ['呼吸系统症状', '循环系统症状', '消化系统症状', '泌尿系统症状', '血液系统症状', \
                    '内分泌代谢症状', '神经系统症状', '生殖系统症状', '运动系统症状', '传染病史', '其他']:
                jbs[k1] = '无' if jbs[k1] == '' else jbs[k1]

            # 手术外伤史
            sss = record['入院记录']['手术外伤史']
            for k1 in ['手术名称及时间', '外伤史', '外伤情况及时间']:
                sss[k1] = '无' if sss[k1] == '' else sss[k1]

            # 输血史
            sxs = record['入院记录']['输血史']
            for k1 in ['血型', '输血时间', '输血不良反应', '临床表现']:
                sxs[k1] = '无' if sxs[k1] == '' else sxs[k1]

            # 药物过敏史
            ywgm = record['入院记录']['药物过敏史']
            for k1 in ['药物过敏史', '过敏药物名称', '临床表现']:
                ywgm[k1] = '无' if ywgm[k1] == '' else ywgm[k1]

            # 个人史
            grs = record['入院记录']['个人史']
            for k1 in ['经常留居地', '地方病地方居住史', '吸烟史', '戒烟时间', '饮酒史', '戒酒时间', '毒品接触史', '毒品名称']:
                grs[k1] = '无' if grs[k1] == '' else grs[k1]

            # 婚育史
            hys = record['入院记录']['婚育史']
            for k1 in ['婚育史', '结婚年龄', '妊娠', '产', '自然生产', '手术生产', '自然流产', '人工流产', '早产', '死产', '引产', '配偶健康状况']:
                hys[k1] = '未知' if hys[k1] == '' else hys[k1]

            results.append(record)

    return results


def write_result(file_path, data):
    """
    将Json数据写入文件
    """
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=1, separators=(',', ':'), ensure_ascii=False))


keys = load_keys(r'data/人机大赛/labeled_ind_192.txt', with_head=False)
keys = ['%s_%s' % (e[0], e[1]) for e in keys]
# 加载json数据
json_data = ''
with open(r'data/人机大赛/汇总结果_231.json') as f:
    json_data = json.load(f, strict=False)
    json_data = filter_records(json_data, keys)
    write_result(r'data/人机大赛/汇总结果_192.json', json_data)
