import os
import torchvision
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.utils import save_image

class Datasets(Dataset):
    def __init__(self,path):
        self.path = path
        self.imgpath = os.path.join(path, "images")
        self.labelpath = os.path.join(path, "masks")
        self.imglist=(os.listdir(self.imgpath))
        self.imglist.sort(key=lambda x: int(x[4:8]))
        self.labellist= os.listdir(self.labelpath)
        self.labellist.sort(key=lambda x: int(x[4:8]))
        self.trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    def __len__(self):
        return len(self.imglist)
    def __trans__(self,img,size):
        h,w=img.shape[:2]
        scale=min(size/h,size/w)
        h=int(h*scale)
        w=int(w*scale)
        img=cv.resize(img,(w,h),interpolation=cv.INTER_CUBIC)
        top = (size - h) // 2
        left = (size- w) // 2
        bottom = size- h - top
        right = size - w - left
        new_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self,index):
        imgname=self.imglist[index]
        labelname=self.labellist[index]
        img_o=cv.imread(os.path.join(self.imgpath,imgname))
        img_l=cv.imread(os.path.join(self.labelpath,labelname))
        #医学图像无彩色，RGB相同，可转为灰度图，节省内存和计算
        img_o=cv.cvtColor(img_o,cv.COLOR_BGR2GRAY)
        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        img_o=self.__trans__(img_o,256)
        img_l = self.__trans__(img_l, 256)
        return self.trans(img_o),self.trans(img_l)
