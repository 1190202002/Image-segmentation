import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

#2维灰度直方图
def Hist2d(image,n):
    if n%2==0: return False
    kernel = np.zeros((n,n))+1/(n*n-1)
    mid=(int)(n/2)
    kernel[mid,mid]=0
    imgarea=cv2.filter2D(image,-1,kernel)
    img2d=cv2.merge([image,imgarea])
    hist2d=cv2.calcHist([img2d],[0,1],None,[256,256],[0,256,0,256])
    P=hist2d/(image.shape[0]*image.shape[1])
    return P,imgarea
#二维最大熵
def Entropy2d(P):
    h=P.shape[0]
    maxEntropy=0
    besti=0
    bestj=0
    lnp=np.log(P)
    lnp[np.isinf(lnp)]=0
    H =-np.sum(lnp*P)
    for i in range(h):
        p1=0
        hst=0
        for j in range(h):
            p1=p1+np.sum(P[0:i+1,j])
            if p1==0:continue
            hst=hst-np.sum(P[0:i+1,j]*lnp[0:i+1,j])
            Hsta=hst/p1+np.log(p1)
            Hstb=(H-hst)/(1-p1)+np.log(1-p1)
            if (Hsta+Hstb)>maxEntropy:
                maxEntropy=(Hsta+Hstb)
                besti = i
                bestj = j
    r, res = cv2.threshold(img, besti, 255, cv2.THRESH_BINARY)
    r2, res2 = cv2.threshold(imgarea, bestj, 255, cv2.THRESH_BINARY)
    cv2.imshow('Entropy2d',res|res2)
#1维最大熵
def Entropy(P):
    h=P.shape[0]
    maxEntropy=0
    besti=0
    lnp=np.log(P)
    lnp[np.isinf(lnp)]=0
    for i in range(h):
            p1=-np.sum(P[0:i+1]*lnp[0:i+1])
            p2 = -np.sum(P[i + 1:h] * lnp[i + 1:h])
            if (p1+p2)>maxEntropy:
                maxEntropy=(p1+p2)
                besti = i
    r, res = cv2.threshold(img, besti, 255, cv2.THRESH_BINARY)
    cv2.imshow('Entropy',res)

img = cv2.imread('BUS Cases/BUS/images/case0007.png',0)
cv2.imshow('Original', img)

hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.show()
Entropy(hist/(img.shape[0]*img.shape[1]))

hist2d,imgarea=Hist2d(img,3)
plt.imshow(hist2d)
plt.show()
Entropy2d(hist2d)






#最大类间方差法
ret,binary=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('OTSU', binary)

cv2.waitKey(0)





