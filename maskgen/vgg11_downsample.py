from pdb import set_trace
from re import S

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vgg11_bn

# cfg = [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 256]

# tile_size: 16
# 480p: 480x720, 30x45
# 720p: 720x1280, 45x80


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.batch_norm = True
        self.model = vgg11_bn(pretrained=True)
        # self.convs = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(2),
        # )
        self.deconvs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(64, 64, kernel_size=5, padding=0, stride=5),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(2),
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=5, mode='bilinear'),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2)
        )
        self.t = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.features[3] = nn.MaxPool2d(
            kernel_size=5, stride=5, padding=0
        )
        self.model.features[7] = nn.MaxPool2d(
            kernel_size=4, stride=4, padding=0
        )

    # def clip(self, x):
    #     x = torch.where(x<0, torch.zeros_like(x), x)
    #     x = torch.where(x>1, torch.ones_like(x), x)
    #     return x

    def forward(self, x):
        x = torch.cat(
            [self.t(_[0, :, :, :])[None, :, :, :] for _ in x.split(1)]
        )
        # skip the last max pooling layer
        x = self.model.features[:28](x)

        # 5x upsample
        x = self.deconvs(x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

