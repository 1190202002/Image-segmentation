import cv2 as cv
import numpy as np
img=cv.imread('BUS Cases/BUS/images/case0001.png')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
_,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('Binary',binary)

kernel=np.ones((7,7),np.uint8)
opening=cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=5)
# opening=cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel,iterations=2)
kernel=np.ones((3,3),np.uint8)
cv.imshow('opening',opening)
sure_bg=cv.dilate(opening,kernel,iterations=3)
sure_fg=cv.erode(opening,kernel,iterations=3)
cv.imshow('sure_bg',sure_bg)
# distance=cv.distanceTransform(opening,cv.DIST_L2,5)
# r,sure_fg=cv.threshold(distance,0.7*distance.max(),255,0)
cv.imshow('sure_fg',sure_fg)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow('unknown',unknown)
ret, markers = cv.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers+1
# 现在让所有的未知区域为0
markers[unknown==255] = 0
markers=cv.watershed(img,markers)
img[markers==-1]=[255,0,0]
cv.imshow("watershed",img)
cv.waitKey(0)