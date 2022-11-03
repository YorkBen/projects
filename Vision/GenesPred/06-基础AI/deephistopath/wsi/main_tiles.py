import slide
import util
import filter
import tiles
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 14330458600000000

# tiles.summary_and_tiles(2, display=False, save_summary=True, save_data=False, save_top_tiles=False)
tiles.singleprocess_filtered_images_to_tiles(display=False, save_summary=False, save_data=True, save_top_tiles=True,
                                           html=False, image_num_list=None)
# tiles.multiprocess_filtered_images_to_tiles()