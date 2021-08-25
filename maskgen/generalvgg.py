from pdb import set_trace

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# cfg = [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 256]

# tile_size: 16
# 480p: 480x720, 30x45
# 720p: 720x1280, 45x80

name2model = {
    "vgg11": vgg11_bn,
    "vgg13": vgg13_bn,
    "vgg16": vgg16_bn,
    "vgg19": vgg19_bn,
}


class FCN(nn.Module):
    def __init__(self, name):
        super(FCN, self).__init__()
        self.batch_norm = True
        self.model = name2model[name](pretrained=True)
        self.convs = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
        )
        self.t = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for param in self.model.features.parameters():
            param.requires_grad = False

    # def clip(self, x):
    #     x = torch.where(x<0, torch.zeros_like(x), x)
    #     x = torch.where(x>1, torch.ones_like(x), x)
    #     return x

    def forward(self, x):

        x = torch.cat(
            [self.t(_[0, :, :, :])[None, :, :, :] for _ in x.split(1)]
        )
        return self.convs(self.maxpool(self.model.features[:15](x)))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

