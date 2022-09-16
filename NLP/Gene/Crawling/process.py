# coding:utf-8

from util import *
import re
import argparse

base_url = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='


def process_page(soup):
    results = []
    section = soup.find(name='section', id='aliases_descriptions')
    if section:
        for subsec in section.find_all(name='div', attrs={'class', 'gc-subsection'}):
            for li in subsec.find_all(name="li"):
                for span in li.find_all(name='span', attrs={'class': 'aliasMainName'}) + \
                    li.find_all(name='span', attrs={'itemprop': 'description'}) + \
                    li.find_all(name='span', attrs={'itemprop': 'alternateName'}):
                    results.append(span.get_text())
    return results

def run(input, output):
    start_idx = 0
    with open(output, 'r') as f_out:
        start_idx = len(list(f_out.readlines()))
    print('start idx: %s' % start_idx)

    results = []
    with open(input) as f:
        for line in f.readlines()[start_idx:]:
            arr = line.strip().split('	')
            print('processing %s' % (base_url + arr[0]))
            driver, soup = get_js_page_object(base_url + arr[0])
            results.append((arr[0], list(set([arr[1]] + process_page(soup)))))
            print(results[-1])
            start_idx = start_idx + 1
            if start_idx % 10 == 0:
                with open(output, 'a+') as f2:
                    for r in results:
                        f2.write('%s	%s\n' % (r[0], '	'.join(r[1])))
                results = []

def process_one(gene_id='MIR6859-1'):
    driver, soup = get_js_page_object(base_url + gene_id)
    result = process_page(soup)
    print(result)

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', type=str, default='gene_names.txt', help='gene_names.txt')
    parser.add_argument('-o', type=str, default='gene_names_out.txt', help='gene_names_out.txt')
    args = parser.parse_args()

    input = args.i
    output = args.o
    if not os.path.exists('%s' % input):
        print('%s not exists' % input)
        exit()

    results = run(input, output)
    # process_one()
