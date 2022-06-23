import cv2
import os
import sys

target_fps = 20

def video_to_photo(video_path, video_img_folder):
    # create video img folder
    if not os.path.exists(video_img_folder):
        os.makedirs(video_img_folder)

    # read video
    cap = cv2.VideoCapture(video_path)

    # The number of videos cropped from the video
    count = 0
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # choose_rate = target_fps / fps
    # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # whether it is opened normally
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if ret == True:
            count += 1
            # if count <= num_frames:
            save_root = os.path.join(video_img_folder, str(count) + '.jpg')
            print("writing " + save_root + " ...")
            cv2.imwrite(save_root, frame)
        cv2.waitKey(1)
    cap.release()


if __name__ == '__main__':
    video_folder = r'data\input\video'
    img_folder = r'data\input\imgs'
    # for filename in os.listdir(video_folder):
    for filename in ['2008.mp4', '2010.mp4', '2016.mp4', '2019.mp4', '3006.mp4']:
        print(filename)
        video_path = os.path.join(video_folder, filename)
        video_img_folder = os.path.join(img_folder, filename.replace('.mp4', ''))
        video_to_photo(video_path, video_img_folder)






    # video_to_photo()
