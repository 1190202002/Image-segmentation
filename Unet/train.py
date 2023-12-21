import torch
import os
import unet
import dataset
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Trainer:
    def __init__(self,path,model,model_copy):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=unet.UNet().to(self.device)
        self.opt=torch.optim.Adam(self.net.parameters())
        self.loss_func=nn.BCELoss()
        self.loader=DataLoader(dataset.Datasets(path),batch_size=3,shuffle=True,num_workers=8)

        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print("Model loaded")
        else:
            print("No model found")
    def train(self,stop_value):
        epoch=1
        loss=1
        while True:
            for inputs,labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value},loss:{loss}",ascii=True, total=len(self.loader)):
                inputs,labels=inputs.to(self.device),labels.to(self.device)
                out=self.net(inputs)
                loss=self.loss_func(out,labels)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            torch.save(self.net.state_dict(),self.model)
            if epoch % 25 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1
if __name__ == '__main__':
    t = Trainer("../BUS Cases/BUS", "./model.plt", "./model_{}_{}.plt")
    t.train(100)





