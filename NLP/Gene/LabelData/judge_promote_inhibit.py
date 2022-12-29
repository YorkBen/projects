import re

def read_gene_data_file(file_path):
    data = []
    whole_line = ''
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line[:-1]
            whole_line = whole_line + line
            if not line.endswith(' '):
                data.append(whole_line)
                whole_line = ''
        data.append(whole_line)
    return data

def sel_item(gene_data):
    result, item = [], []
    start_nbr, i = 1, 0
    while i < len(gene_data):
        if gene_data[i].startswith('%s.' % start_nbr):
#             print('start nbr: %s match!' % start_nbr)
            if len(item) > 0:
                # result.append(item)
                result.append('%s\n' % '\\n'.join(item))
                item = []

            start_nbr = start_nbr + 1
        item.append(gene_data[i])
        i = i + 1

    if len(item) > 0:
        # result.append(item)
        result.append('%s\n' % '\\n'.join(item))

    print('item length: %s' % len(result))
    return result


def load_regex(file_path):
    inhibits = []
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            for w in re.split('[，,]', line.strip()):
                w = w.strip()
                if w != '':
                    inhibits.append(w)
    inhibits = list(set(inhibits))
    inhibits_regex = '(' + ')|('.join(inhibits) + ')'

    return inhibits_regex


if __name__ == "__main__":
    inhibits_regex = load_regex('../Data/20221207/inhibits-关键词 - 2.0.txt')
    promotes_regex = load_regex('../Data/20221207/promote-关键词 - 2.0.txt')

    passages = sel_item(read_gene_data_file('../Data/20221207/igf1+cancer19770201-20131205.txt'))
    passages += sel_item(read_gene_data_file('../Data/20221207/igf2+cancer19701101-20221205.txt'))
    passages += sel_item(read_gene_data_file('../Data/20221207/wnt+cancer19870814-20140814.txt'))
    passages += sel_item(read_gene_data_file('../Data/20221207/wnt+cancer20140815-20201101.txt'))
    passages += sel_item(read_gene_data_file('../Data/20221207/wnt+cancer20201102-20221205.txt'))

    inhibits_passages = []
    promotes_passages = []
    uncertain_passages = []
    zero_passages = []
    for idx, passage in enumerate(passages):
        if (idx + 1) % 100 == 0:
            print('processing idx: %d' % (idx+1))

        inhibit_ct, promote_ct = 0, 0
        for match in re.finditer(inhibits_regex, passage):
            if match is not None:
                inhibit_ct += 1

        for match in re.finditer(promotes_regex, passage):
            if match is not None:
                promote_ct += 1

        if inhibit_ct == 0 and promote_ct == 0:
            zero_passages.append(passage)
        elif inhibit_ct > promote_ct:
            inhibits_passages.append(passage)
        elif inhibit_ct < promote_ct:
            promotes_passages.append(passage)
        else:
            uncertain_passages.append(passage)

    print(len(zero_passages), len(inhibits_passages), len(promotes_passages), len(uncertain_passages))
    with open('../Data/20221207/zero_data.text', 'w') as f:
        for item in zero_passages:
            f.write(item)

    with open('../Data/20221207/inhibits_data.text', 'w') as f:
        for item in inhibits_passages:
            f.write(item)

    with open('../Data/20221207/promotes_data.text', 'w') as f:
        for item in promotes_passages:
            f.write(item)

    with open('../Data/20221207/uncertain_data.text', 'w') as f:
        for item in uncertain_passages:
            f.write(item)
