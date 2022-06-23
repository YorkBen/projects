import cv2
import numpy as np

# cap = cv2.VideoCapture("data/165321343-1-208.mp4")
cap = cv2.VideoCapture("data/169743465-1-16.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (600, 800))
    rows, cols, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    _, threshold = cv2.threshold(gray_frame, 28, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.line(frame, (x+int(w/2), 0), (x+int(w/2), rows), (0, 255,0), 2)
        cv2.line(frame, (0,y+int(h/2)), (cols, y+int(h/2)), (0,255,0), 2)
        break

    cv2.imshow("gray_frame", gray_frame)
    cv2.imshow("threshold", threshold)
    cv2.imshow("frame", frame)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cv2.destroyAllWindows()
