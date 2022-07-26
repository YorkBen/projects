import json

json_data = ''
with open(r'汇总结果.json') as f:
    json_data = json.load(f, strict=False)

# print(len(json_data))

for r in json_data:
    if "出院记录" in r:
        print(r["医保编号"], r["出院记录"]['出院诊断'])
    else:
        print(r["医保编号"])
