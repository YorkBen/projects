import PyPDF2
import pdfplumber
import re
import os

def process_page(text):
    """
    处理一页
    """
    # print(text)
    # print('')

    results = []
    tbl_start = False
    id, date, sjmd, bbzl = '', '', '', ''
    for line in text.split('\n'):
        if '病人号ID:' in line:
            id = line.split('病人号ID:')[1]
        if '报告单ID:' in line:
            date = line.split('报告单ID:')[1].split('-')[0]
        if '标本种类:' in line:
            tmp = line.split('标本种类:')[1]
            arr = tmp.split(' ')
            bbzl = arr[0].strip()
        if '送检目的:' in line:
            tmp = line.split('送检目的:')[1]
            arr = tmp.split(' ')
            sjmd = arr[0].strip()
        if line.startswith('序号'):
            tbl_start = True
        if tbl_start:
            if line.startswith('备注'):
                break
            elif re.match('[1-9].*', line):
                if '■特别说明' in line:
                    continue

                if len(line.split(' ')) < 6:
                    continue

                arr = line.split(' ')
                # print('%s,%s,%s,%s,%s,%s,%s,%s' % (id, no, sjmd, bbzl, arr[1], arr[3], arr[-2], arr[-1]))
                results.append('%s,%s,%s,%s,%s,%s,%s,%s' % (id, date, sjmd, bbzl, arr[1], arr[3], arr[-2], arr[-1]))

    return results
    # print('病人号ID: %s' % id)
    # print('报告单ID: %s' % no)
    # print('送检目的: %s' % sjmd)
    # print('标本种类: %s' % bbzl)

def write_result(results, file_path):
    if len(results) > 0:
        with open(file_path, 'a+') as f:
            for line in results:
                f.write(line + '\n')



if __name__ == '__main__':
    out_file = r'D:\projects\NLP\MedicalRecord\FeatureExtraction\data\202207人机大赛\检验数据.txt'
    # with open(out_file, 'a+') as f:
    #     f.write('medicalrecordno,流水号,组套名称,标本,项目,检验值,单位,参考值范围\n')

    in_folder = r'D:\项目资料\病历辅助诊断\人机大赛数据集'
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            if name.endswith('.pdf'):
                file_path = os.path.join(root, name)
                # print('processing %s' % file_path)
                if '检验' in name:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            write_result(process_page(page.extract_text()), out_file)
                else:
                    print('%s not processed!' % file_path)

#
