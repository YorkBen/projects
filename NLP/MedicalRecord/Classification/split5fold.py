from Lib.DataLoader import DataLoader

if __name__ == "__main__":
    split_num=5
    file_name = '疾病诊断拟合_全特征'
    separator='	'

    dl = DataLoader()
    lines = dl.load_data_lines(r'data/%s.txt' % file_name, num_fields=171, separator=separator, skip_title=True, shuffle=True)
    lines_arr = dl.split_n_folds(lines, cls_col=2, split_num=split_num)
    for i in range(split_num):
        with open(r'data/%s_%s.txt' % (file_name, (i+1)), 'w') as f:
            for line in lines_arr[i]:
                f.write('%s\n' % separator.join(line))
