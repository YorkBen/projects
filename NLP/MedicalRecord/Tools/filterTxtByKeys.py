keys = []
with open(r'疾病诊断拟合_全特征_1.txt') as f:
    for line in f.readlines():
        arr = line.strip().split('	')
        keys.append('%s_%s' % (arr[0], arr[1]))

# name = '疾病诊断拟合_临床合成'
name = '疾病诊断拟合_临床实验室合成'
data1, data2345 = [], []
with open('%s.txt' % name) as f:
    for line in f.readlines():
        arr = line.strip().split('	')
        key = '%s_%s' % (arr[0], arr[1])
        if key in keys:
            data1.append(line)
        else:
            data2345.append(line)

with open('%s1.txt' % name, 'w') as f:
    for line in data1:
        f.write(line)

with open('%s2345.txt' % name, 'w') as f:
    for line in data2345:
        f.write(line)
