import logging

import coloredlogs
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from utils.timer import Timer

x = Image.open("visdrone/videos/vis_171/%010d.png" % 1)
x = T.ToTensor()(x)[None, :, :, :]
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
logger = logging.getLogger("T")
coloredlogs.install(
    fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
    level="INFO",
)

for i in range(100):
    print(i)
    with Timer("run", logger):
        model(x)
