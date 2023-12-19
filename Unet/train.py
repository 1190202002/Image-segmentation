import torch
import os
import unet
import dataset
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Trainer:
    def __init__(self,path,model,model_copy,img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path =img_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=unet.UNet().to(self.device)
        self.opt=torch.optim.Adam(self.net.parameters())
        self.loss_func=nn.BCELoss()
        self.loader=DataLoader(dataset.Datasets(path),batch_size=1,shuffle=True,num_workers=4)

        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print("Model loaded")
        else:
            print("No model found")
        os.makedirs(self.img_save_path,exist_ok=True)
    def train(self,stop_value):
        epoch=1
        while True:
            for inputs,labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",ascii=True, total=len(self.loader)):
                inputs,labels=inputs.to(self.device),labels.to(self.device)
                out=self.net(inputs)
                loss=self.loss_func(out,labels)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                x=inputs[0]
                x_=out[0]
                y=labels[0]
                img=torch.stack([x,x_,y],dim=0)
                save_image(img.cpu(),os.path.join(self.img_save_path,"epoch_"+str(epoch)+".png"))
            torch.save(self.net.state_dict(),self.model)

            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1
if __name__ == '__main__':
	# 路径改一下
    t = Trainer("/home/yf/PycharmProjects/imagesegment/BUS Cases/BUS", './model.plt', r'./model_{}_{}.plt', img_save_path=r'./train_img')
    t.train(300)





