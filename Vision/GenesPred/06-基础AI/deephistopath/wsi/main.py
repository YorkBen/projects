import slide
import util
import filter
import cv2

# img_path = slide.get_training_image_path(2)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# cv2.imwrite('gray.jpg', grayscale)
# print(img_path)

filter.apply_filters_to_image(2, display=False, save=True)