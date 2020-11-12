
import torch
import torch.nn as nn

cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256]

class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.batch_norm = True
        self.model = self._make_layers(cfg)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 6
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]
        return nn.Sequential(*layers)

    def clip(self, x):
        x = torch.where(x<0, torch.zeros_like(x), x)
        x = torch.where(x>1, torch.ones_like(x), x)
        return x

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))