import torch
import os
import unet
import dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Test:
    def __init__(self,path,model,img_save_path):
        self.path = path
        self.model = model
        self.img_save_path =img_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=unet.UNet().to(self.device)
        self.loader=DataLoader(dataset.Datasets(path),batch_size=1,shuffle=True,num_workers=8)
        self.net.load_state_dict(torch.load(model))
        os.makedirs(self.img_save_path,exist_ok=True)
    @torch.no_grad()
    def test(self):
            i=1
            for inputs,labels in self.loader:
                inputs,labels=inputs.to(self.device),labels.to(self.device)
                out=self.net(inputs)
                x=inputs[0]
                x_=out[0]
                y=labels[0]
                img=torch.stack([x,x_,y],dim=0)
                save_image(img.cpu(),os.path.join(self.img_save_path,"test"+str(i)+".png"))
                i+=1

if __name__ == '__main__':
    t = Test("/home/yf/PycharmProjects/imagesegment/BUS Cases/test",'./model.plt', img_save_path=r'./result')
    t.test()





