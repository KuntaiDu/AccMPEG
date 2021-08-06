"""
    The factory to build DNN according to 1). the name or 2). the yml file
"""

from .coco_model import COCO_Model
from .fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from .fcn_resnet50 import FCN_ResNet50
from .segmentation import Segmentation
from .efficient_det.interface import EfficientDet


class DNN_Factory:
    def __init__(self):
        self.name2model = {
            "FasterRCNN_ResNet50_FPN": FasterRCNN_ResNet50_FPN,
            "EfficientDet": EfficientDet,
        }

    def get_model(self, name):

        if name in self.name2model:
            return self.name2model[name]()
        elif "Segmentation" in name:
            return Segmentation(name)
        elif name == "fcn_resnet50":
            return fcn_resnet50()
        else:
            assert "yaml" in name
            return COCO_Model(name)
