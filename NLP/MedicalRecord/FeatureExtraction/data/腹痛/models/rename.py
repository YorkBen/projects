import os
import shutil

for file in os.listdir('.'):
    if file.endswith('.pth') and '_' in file:
        new_name = file.split('_')[0] + '.pth'
        shutil.move(file, new_name)
