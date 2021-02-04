"""
    The factory to build DNN according to 1). the name or 2). the yml file
"""

from .coco_model import COCO_Model
from .fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from .fcn_resnet101 import FCN_ResNet101


class DNN_Factory:
    def __init__(self):
        self.name2model = {
            "FasterRCNN_ResNet50_FPN": FasterRCNN_ResNet50_FPN,
            "FCN_ResNet101": FCN_ResNet101,
        }
        pass

    def get_model(self, name):

        if name in self.name2model:
            return self.name2model[name]()
        else:
            assert "yaml" in name
            return COCO_Model(name)
