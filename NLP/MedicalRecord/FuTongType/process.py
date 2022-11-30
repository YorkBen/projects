import re

results = []
with open('feature_text.txt', encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()
        results.extend(re.split('[，。]', line))

results = list(set(results))

with open('feature_word.txt', "w", encoding="utf-8") as f:
    for r in results:
        f.write('%s\n' % r)
