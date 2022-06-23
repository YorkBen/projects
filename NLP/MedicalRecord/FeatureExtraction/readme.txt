文件说明：
  1. process_medical_record.py：过滤筛选病历记录，减少后期运算量，因为结构化需要耗时较长。
  2. process_inspect_all.py：处理所有的实验室数据，从里面选出需要的记录
  3. process_yizhu_all.py：处理所有的医嘱数据，从里面选出需要的记录。
  4. process_others.py：处理超声，放射，病理等其它数据。
  5. split_record_bytype.py：病历数据按照类型划分统计。
  6. run.py：主程序，将所有数据连接起来，合并成最终json
  7. jsontoexcel.py：将json转化为excel的程序。
