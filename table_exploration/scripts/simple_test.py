import numpy as np
import cv2
import imutils

image = cv2.imread("red_gazebo.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# red color boundaries [B, G, R]
min_red = np.array([0, 10, 120]) 
max_red = np.array([15, 255, 255]) 
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 


mask_r = cv2.inRange(hsv, min_red, max_red) 
output = cv2.bitwise_and(image, image, mask=mask_r)
ret,thresh = cv2.threshold(mask_r, 20, 255, 9)
contours = cv2.findContours(mask_r.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(contours)
c = max(cnts, key = cv2.contourArea)
marker= cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
draw_im = cv2.drawContours(output, [box], -1, (0, 255, 0), 2)

cv2.imshow("BOX",output)

cv2.imshow("bitwise",np.hstack([image, output]))
cv2.waitKey(0)

