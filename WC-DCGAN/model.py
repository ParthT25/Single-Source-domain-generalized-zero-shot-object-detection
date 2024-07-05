import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x 2054 x 7 x 7
            nn.Conv2d(2054, 1024, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            self._block(1024, 512, 2, 1, 0),
            self._block(512, 256, 2, 1, 0),
            self._block(256, 128, 2, 1, 0),
            self._block(128, 64, 2, 1, 0),
            nn.Conv2d(64, 1, kernel_size=2, stride=1, padding=0),  # Output: N x 1 x 1 x 1
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x, labels):
        # Concatenate input image and label embeddings along the channel dimension
        x = torch.cat([x, labels], dim=1)  # N x (C+embed_size) x H x W
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, embed_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x (z_dim + embed_size) x 1 x 1
            self._block(z_dim + embed_size, 64, 2, 1, 0),  # N x 64 x 2 x 2
            self._block(64, 128, 2, 1, 0),  # N x 128 x 3 x 3
            self._block(128, 256, 2, 1, 0),  # N x 256 x 4 x 4
            self._block(256, 512, 2, 1, 0),  # N x 512 x 5 x 5
            self._block(512, 1024, 2, 1, 0),  # N x 1024 x 6 x 6
            nn.ConvTranspose2d(
                1024, 2048, kernel_size=2, stride=1, padding=0
            ),  # Output: N x 2048 x 7 x 7
            nn.ReLU(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
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
    
    def forward(self, x, labels):
        # Expand label embeddings to match dimensions of the input noise
        embedding = labels.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)
