import cv2
import numpy as np

# Load image, grayscale, Gaussian blur, adaptive threshold
image = cv2.imread(r'tesseract\imgs4\04.jpg')

height, width = image.shape[:2]
# image = cv2.resize(image, (width//2, height//2), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)
kernel = np.ones((5,5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
# blur = cv2.GaussianBlur(gray, (9,9), 0)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)
# Dilate to combine adjacent text contours
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
# dilate = cv2.dilate(thresh, kernel, iterations=4)

# d = (dilate > 0)
# edges[d] = 255
# edges[~d] = 0
# # Find contours, highlight text areas, and extract ROIs
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# ROI_number = 0
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area > 10000:
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
#         # ROI = image[y:y+h, x:x+w]
#         # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#         # ROI_number += 1

cv2.imshow('edges', edges)
cv2.imshow('image', image)
cv2.waitKey()
