import os
import glob
import shutil
from tqdm import tqdm

path = r"/root/data1/zhangkuo/DX0810-tmp"
files = glob.glob(path + "/*/*/*.svs")
new_path = r"/root/data1/zhangkuo/DX0810"
for file in tqdm(files):
    fname = os.path.basename(file)

    shutil.move(file, os.path.join(new_path, fname))