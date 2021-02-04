from collections import OrderedDict

import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from . import carn


class CARN:
    def __init__(self, upscale=2):
        self.net = carn.Net(multi_scale=True, group=1)
        self.upscale = upscale

        state_dict = torch.load("/tank/kuntai/code/video-compression/dnn/CARN/carn.pth")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            # name = k[7:] # remove "module."
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        self.net.cuda()

    def __call__(self, image):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            image = self.net(image, self.upscale)
        # print(image)
        # input()
        return image
