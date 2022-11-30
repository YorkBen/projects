import sys

sys.path.append('Lib')
from CommonSplit import CommonSplit
from TextStructral import TextStructral

def get_mr_no(line):
    """
    获取医保号码
    """
    return line.split('||')[0].strip()

if __name__ == "__main__":
    ts = TextStructral()
    ps = CommonSplit(template_file=r'data/template/all.json')

    data_type, postfix = '黄疸', 'all'

    # 入院记录
    # records = ts.load_records('data/%s/tmp/mr_ry_%s.txt' % (data_type, postfix), ['1000012'])
    # for idx, record in enumerate(records):
    #     if len(record) < 2 or len(record[1]) < 100:
    #         continue
    #     result = ps.process(record[1], "root")
    #     # if '一般信息' in result['root'] and '床号' in result['root']['一般信息']:
    #     #     print(result['root']['一般信息']['住院号'], result['root']['一般信息']['床号'])
    #     print(result)
    #     print(record)

    # # 出院记录
    # records = ts.load_records('data/%s/tmp/mr_cy_%s.txt' % (data_type, postfix), ['D012871', 'D013461', 'D014381'])
    # for idx, record in enumerate(records):
    #     if len(record) < 2 or len(record[1]) < 100:
    #         continue
    #     result = ps.process(record[1], "root")
    #     # if '一般信息' in result['root'] and '床号' in result['root']['一般信息']:
    #     #     print(idx, result['root']['一般信息']['住院号'], result['root']['一般信息']['床号'])
    #     print(result)


    # 首次病程
    records = ts.load_records('data/%s/tmp/mr_sc_%s.txt' % (data_type, postfix), ['1000012'])
    for idx, record in enumerate(records):
        if len(record) < 2 or len(record[1]) < 100:
            continue
        result = ps.process(record[1], "首次病程记录")
        # if '首次病程记录' in result and '床号' in result['首次病程记录']:
        #     print(idx, result['首次病程记录']['住院号'], result['首次病程记录']['床号'])
        print(result)
        print(record)

    # # 日常病程记录
    # records = ts.load_records('data/%s/tmp/mr_rc_%s.txt' % (data_type, postfix))
    # for idx, record in enumerate(records):
    #     if len(record) < 2 or len(record[1]) < 100:
    #         continue
    #
    #     result = ps.process(record[1][:100], "日常病程记录")
    #     if '日常病程记录' in result and '床号' in result['日常病程记录']:
    #         if '住院号' in result['日常病程记录']:
    #             mr_no = result['日常病程记录']['住院号']
    #         else:
    #             mr_no = get_mr_no(record[0])
    #         print(idx, mr_no, result['日常病程记录']['床号'])
