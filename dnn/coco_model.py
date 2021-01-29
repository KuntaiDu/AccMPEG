import logging

import detectron2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from .dnn import DNN


class COCO_Model(DNN):
    def __init__(self, name):

        self.name = name

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(name))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
        self.model = DefaultPredictor(self.cfg)
        self.model.eval()

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.class_ids = [1, 2, 3, 4, 6, 7, 8]

    def cpu(self):

        self.model.cpu()
        self.logger.info(f"Place %s on CPU.", self.name)

    def cuda(self):

        self.model.cuda()
        self.logger.info(f"Place %s on GPU.", self.name)

    def inference(self, img):

        return self.model(img)

    def calc_loss(self, img, result):

        # convert result to target
        pass

