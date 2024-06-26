import torch
import torch.nn as nn
import torch.nn.functional as F 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            #N x channels x 14 x 14
            nn.Conv2d(
                2054,1024,kernel_size=2,stride=1,padding=0
            ),
            # 3 x 3
            nn.LeakyReLU(0.2),
            self._block(1024,512,2,1,0), # 2x2
            self._block(512,256,2,1,0),
            self._block(256,128,2,1,0),
            self._block(128,64,2,1,0),
            # self._block(64,1,2,1,0)
            nn.Conv2d(64,1,kernel_size=2,stride=1,padding=0),#1 x 1
            # nn.Sigmoid()
        )
    def _block(self,in_channels,out_channels,kernal_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernal_size,
                stride,
                padding,
                bias=True
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x,labels):
        # embedding = self.embed(labels).view(labels.shape[0],1,self.img_size,self.img_size)
        # print(x.shape,labels.shape)
        x = torch.cat([x,labels],dim=1) # N x C x H x W
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self,z_dim,embed_size):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            #input Nxz_dimx1x1
            self._block(z_dim+embed_size,64,2,1,0), # N x fx16 x 4 x 4
            self._block(64,128,2,1,0), # 4 x 4
            self._block(128,256,2,1,0),
            self._block(256,512,2,1,0),
            self._block(512,1024,2,1,0),
            nn.ConvTranspose2d(
                1024,2048,kernel_size=2,stride=1,padding=0
            ),
            nn.ReLU(),
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self,x,labels):
        # embedding = labels.squeeze(1)
        embedding = labels.unsqueeze(2).unsqueeze(3)
        # print("gen",x.shape,embedding.shape)
        x = torch.cat([x,embedding],dim=1)
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0,0.02)
