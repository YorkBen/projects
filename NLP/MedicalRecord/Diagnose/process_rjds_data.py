
data_file1 = r'data/人机大赛_数据库导出.txt'
data_file2 = r'data/人机_临床.txt'

data_dict = {}
with open(data_file2) as f:
    for line in f.readlines()[1:]:
        arr = line.strip().split('	')
        data_dict[arr[0]] = arr[:121]


data = []
with open(data_file1) as f:
    for line in f.readlines():
        arr = line[:-1].split('\t')
        mrno = arr[0][:-2]
        if len(arr) == 1 or arr[1] == '':
            feature_lab = ['2'] * 19
        else:
            feature_lab = [f[0] if isinstance(f, tuple) else f for f in eval(arr[1])]
        feature_lab = [str(e) for e in feature_lab]

        if len(arr) <= 2 or arr[2] == '':
            feature_insp = ['2'] * 36
        else:
            feature_insp = eval(arr[2])
        feature_insp = [str(e) for e in feature_insp]

        all_feature = data_dict[mrno]
        data.append('%s\t%s\t%s\n' % ('\t'.join(all_feature), '\t'.join(feature_lab), '\t'.join(feature_insp)))

with open(r'data/人机大赛导出全特征.txt', 'w') as f:
    for line in data:
        f.write(line)
