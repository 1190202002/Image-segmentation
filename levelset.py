import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

Image = cv2.imread('BUS Cases/BUS/images/case0001.png')
image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image,(7,7),5)
image = cv2.fastNlMeansDenoising(image, None, 25,7,21)


cv2.imshow('image',image)
img = np.array(image, dtype=np.float64)

# 初始水平集函数
LSF = np.ones(img.shape[:2], img.dtype)
LSF[300:320, 300:320] = -1
LSF=-LSF

# CV函数
def CV(LSF, img, mu, nu, epison, dt):
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * np.arctan(LSF / epison))
    gd= np.gradient(LSF)
    gd_norm=gd/(np.sqrt(gd[0]**2 + gd[1]**2)+ 0.000001)
    Mxx, Nxx = np.gradient(gd_norm[0])
    Nyy, Myy = np.gradient(gd_norm[1])
    cur = Nxx + Nyy
    Length = nu * Drc * cur

    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu * (Lap - cur)

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

    LSF = LSF + dt * (Length +Penalty+ CVterm)
    return LSF


# 模型参数
mu = 1
nu = 0.003 * 255 * 255
iter= 10
epison = 1
dt = 0.1
for i in range(iter):
    LSF = CV(LSF, img, mu, nu, epison, dt)  # 迭代
kernel = np.ones((3, 3), np.uint8)
LSF=cv2.erode(LSF, kernel,iterations=5)
plt.imshow(Image), plt.xticks([]), plt.yticks([])
plt.contour(LSF, [0], colors='r', linewidths=1)
plt.draw(), plt.show(block=False)
cv2.imshow('LSF', LSF)
cv2.waitKey(0)