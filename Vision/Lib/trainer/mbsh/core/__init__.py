import os
import datetime
import random

__author__ = 'gpu'
fine_img_name_len = 30


def cal_file_name(file_name):
    name, file_extension = os.path.splitext(file_name)
    if len(file_name) >= fine_img_name_len:
        sub_name = file_name
    else:

        sub_name = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S_') + str(random.randint(0, 1000000)).zfill(12)
        sub_name += file_extension
        assert len(sub_name) >= fine_img_name_len
    return sub_name
