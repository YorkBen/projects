import cv2
import os
import sys

def video_to_photo(video_path, video_img_folder, target_fps=0):
    # create video img folder
    if not os.path.exists(video_img_folder):
        os.makedirs(video_img_folder)

    # read video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    choose_rate = target_fps / fps
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print('processing video: %s' % video_path)
    print('video frame num: %s' % total_frame_num)
    print('video fps: %s, choose fps: %s, choose rate: %.2f' % (fps, target_fps, choose_rate))
    print('frame should choose: %d' % (choose_rate * total_frame_num))

    # The number of videos cropped from the video
    # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # whether it is opened normally
    total_ct, choose_ct = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if ret == True:
            total_ct = total_ct + 1
            if int(total_ct * choose_rate) - choose_ct == 1:
                choose_ct = choose_ct + 1
                # if count <= num_frames:
                save_root = os.path.join(video_img_folder, str(choose_ct) + '.jpg')
                # print("writing " + save_root + " ...")
                cv2.imwrite(save_root, frame)
                if choose_ct > 0 and (choose_ct % 100) == 0:
                    print('frame choosed: %s' % choose_ct)
        cv2.waitKey(1)
    cap.release()


def cropImg(video_img_folder, crop_img_folder, region, debug=False):
    # create crop img folder
    if not os.path.exists(crop_img_folder):
        os.makedirs(crop_img_folder)

    for file in os.listdir(video_img_folder):
        img = cv2.imread(os.path.join(video_img_folder, file))
        img_ = img[region[0]:region[1], region[2]:region[3]]

        if debug:
            cv2.namedWindow("enhanced", 0);
            cv2.imshow('enhanced', img_)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()

        cv2.imwrite(os.path.join(crop_img_folder, file), img_)



if __name__ == '__main__':
    # base_dir = r'D:\项目资料\息肉假阳性\202208\有息肉肠镜视频'
    base_dir = r'D:\项目资料\息肉假阳性\202208\无息肉肠镜视频'

    video_folder = os.path.join(base_dir, 'videos')
    img_folder = os.path.join(base_dir, 'images')
    crop_img_folder = os.path.join(base_dir, 'images_crop')
    target_fps = 18
    # for filename in os.listdir(video_folder):
    regions = [
        [30,1045, 696,1860],
        [0,950, 465,1560],
        [34,1044, 517,1685]
    ]
    # videos = {'3400-原视频.mp4':0, '3404-原视频.mp4':0, '3406-原视频.mp4':0, '3410-原视频.mp4':0, '3424-原视频.mp4':0, '3428-原视频.mp4':0, '3431-原视频.mp4':0, '3433-原视频.mp4':0, '3436-原视频.mp4':0, '3444-原视频.mp4':0, '3454-原视频.mp4':0, '3455-原视频.mp4':0, '3457-原视频.mp4':0, '3459-原视频.mp4':0, '3460-原视频.mp4':0, '3462-原视频.mp4':0, '3469-原视频.mp4':0, '3471-原视频.mp4':0, '3474-原视频.mp4':0, '3476-原视频.mp4':0, '3477-原视频.mp4':0, '3478-原视频.mp4':0, '3479-原视频.mp4':0, '3480-原视频.mp4':0, '3481-原视频.mp4':0, '3485-原视频.mp4':0, '3488-原视频.mp4':0, '3496-原视频.mp4':0, '3497-原视频.mp4':0, '3499-原视频.mp4':0, '3504-原视频.mp4':0, '3510-原视频.mp4':0, '3513-原视频.mp4':0, '3515-原视频.mp4':0, '3520-原视频.mp4':0, '3526-原视频.mp4':0, '3531-原视频.mp4':0, '3532-原视频.mp4':0, '3533-原视频.mp4':0, '3538-原视频.mp4':0}
    videos = {name:0 for name in os.listdir(video_folder)}
    for filename in list(videos.keys()):
        print('processing: %s' % filename)
        region = regions[videos[filename]]
        video_path = os.path.join(video_folder, filename)
        name = filename.replace('-原视频', '').replace('.mp4', '')
        video_img_folder = os.path.join(img_folder, name)
        video_crop_img_folder = os.path.join(crop_img_folder, name)
        video_to_photo(video_path, video_img_folder, target_fps)
        cropImg(video_img_folder, video_crop_img_folder, region, False)




##
