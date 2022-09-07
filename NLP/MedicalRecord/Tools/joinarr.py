with open('tmp.txt') as f1:
    with open('tmpout.txt', 'w') as f2:
        for line in f1.readlines():
            arr = line.strip().split('	')
            f2.write('[%s]\n' % ','.join(arr))
