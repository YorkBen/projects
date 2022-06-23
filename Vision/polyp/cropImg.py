import cv2
import os
import sys

if __name__ == '__main__':
    img_folder = r'data\input\imgs'
    # for folder in os.listdir(img_folder):
    for folder in ['2010', '2016', '2019']:
        print('processing %s' % folder)
        if 'crop' in folder:
            continue

        sub_folder = os.path.join(img_folder, folder)
        target_folder = os.path.join(img_folder, folder + '_crop')
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        for file in os.listdir(sub_folder):
            img = cv2.imread(os.path.join(sub_folder, file))
            # 视频 '3006', '3450', '3452', '3456', '3466', '3486'
            # img_ = img[30:1045, 695:1860]
            # 视频 '2008'
            # img_ = img[0:950, 465:1560]
            # 视频 '2010', '2016', '2019'
            img_ = img[34:1044, 517:1685]
            # cv2.namedWindow("enhanced", 0);
            # cv2.imshow('enhanced', img_)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # exit()
            cv2.imwrite(os.path.join(target_folder, file), img_)
