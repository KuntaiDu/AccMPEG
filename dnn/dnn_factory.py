"""
    The factory to build DNN according to 1). the name or 2). the yml file
"""

from .coco_model import COCO_Model
from .fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from .detr_resnet101 import Detr_ResNet101
from .segmentation import Segmentation
from dnn.efficient_det.efficientdet_accmpeg import EfficientDet

class DNN_Factory:
    def __init__(self):
        self.name2model = {
            "FasterRCNN_ResNet50_FPN": FasterRCNN_ResNet50_FPN,
            "Detr_ResNet101": Detr_ResNet101,
            "EfficientDet": EfficientDet
        }
        pass

    def get_model(self, name):

        if name in self.name2model:
            return self.name2model[name]()
        elif "Segmentation" in name:
            return Segmentation(name)
        else:
            assert "yaml" in name
            return COCO_Model(name)
