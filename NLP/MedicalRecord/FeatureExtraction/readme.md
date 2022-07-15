文件说明：
  1. select_data_with_mrnos.py：根据医保编号选取各类数据。
  2. split_record_bytype.py：病历数据按照类型划分统计。
  3. run.py：主程序，将所有数据连接起来，合并成最终json
  4. jsontoexcel.py：将json转化为excel的程序。
  5. extractDoc.py：手工导出的病历数据，如入院记录、出院记录、病程记录都是doc格式，代码用于处理这三种Doc文件转化为标准导入格式
  6. extractPdf.py：转化手工导出的检验数据到标准格式。
  7. extractImg.py：转化手工导出的CT、MR等影像学、超声检查图片数据到标准数据格式。
