### 文件说明：
  1. select_data_with_mrnos.py：根据医保编号选取各类数据。
  2. split_record_bytype.py：病历数据按照类型划分统计。
  3. run.py：主程序，将所有数据连接起来，合并成最终json
  4. jsontoexcel.py：将json转化为excel的程序。
  5. extractDoc.py：手工导出的病历数据，如入院记录、出院记录、病程记录都是doc格式，代码用于处理这三种Doc文件转化为标准导入格式
  6. extractPdf.py：转化手工导出的检验数据到标准格式。
  7. extractImg.py：转化手工导出的CT、MR等影像学、超声检查图片数据到标准数据格式。


### 命令说明：
1. 筛选数据：python select_data_with_mrnos.py -t 腹痛 -p 4335
2. 将从信息科拷贝的1611条数据合并到本机数据中：
~/projects/NLP/MedicalRecord/FeatureExtraction/data/腹痛/tmp/从信息科抓取/1611
  bash cattxt.sh 4335
3. 生成汇总数据：python .\run.py -t 腹痛 -p 4335
4. 生成excel：python .\jsonToExcel.py -t 腹痛 -p 4335
5. 生成excel：python .\jsonToExcel_2.py -t 腹痛 -p 4335
