import logging
from pdb import set_trace

import torch
import torch.nn.functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from PIL import Image

from .dnn import DNN


class SSD(DNN):
    def __init__(self):

        self.name = "MobileNet-SSD"

        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd"
        )
        self.utils = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_ssd_processing_utils",
        )
        self.model.cuda()
        self.model.eval()

        self.logger = logging.getLogger(self.name)
        self.class_ids = [0, 1, 2, 3, 5, 6, 7]

    def inference(self, image, detach=False, grad=False):

        _, _, h, w = image.shape
        assert h == 720 and w == 1280

        # resize to COCO model size
        image = F.interpolate(image, (300, 300))
        # rescale
        image = (image * 256 - 128) / 128
        image = image.cuda()

        with torch.enable_grad() if grad else torch.no_grad():
            result = self.utils.decode_results(self.model(image))
            result = result[0]
            result = self.utils.pick_best(result, 0.0)

            bboxes, classes, confidences = result

            bboxes = [
                [left * w, bot * h, right * w, top * h]
                for left, bot, right, top, in bboxes
            ]

        ret = Instances(
            image_size=(h, w),
            pred_boxes=Boxes(bboxes),
            scores=confidences,
            pred_classes=(classes - 1),
        )

        if detach:
            ret = ret.to("cpu")

        return {"instances": ret}

    def filter_result(self, result, args, gt=False, confidence_filter=True):

        scores = result["instances"].scores
        class_ids = result["instances"].pred_classes
        bboxes = result["instances"].pred_boxes

        inds = scores < 0
        for i in self.class_ids:
            inds = inds | (class_ids == i)

        if confidence_filter:
            if gt:
                inds = inds & (scores > args.gt_confidence_threshold)
            else:
                inds = inds & (scores > args.confidence_threshold)

        result["instances"] = result["instances"][inds]

        return result

    def visualize(self, image, result, args):
        # set_trace()
        result = self.filter_result(result, args)
        v = Visualizer(image, MetadataCatalog.get("coco_2017_train"), scale=1)
        out = v.draw_instance_predictions(result["instances"])
        return Image.fromarray(out.get_image(), "RGB")
