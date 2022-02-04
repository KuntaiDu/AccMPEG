import copy
import logging
from copy import deepcopy
from pdb import set_trace

import detectron2
import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import pairwise_iou
from detectron2.structures.keypoints import Keypoints
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import Visualizer
from PIL import Image

from .dnn import DNN

panoptic_segmentation_labels = [
    "things",
    "banner",
    "blanket",
    "bridge",
    "cardboard",
    "counter",
    "curtain",
    "door-stuff",
    "floor-wood",
    "flower",
    "fruit",
    "gravel",
    "house",
    "light",
    "mirror-stuff",
    "net",
    "pillow",
    "platform",
    "playingfield",
    "railroad",
    "river",
    "road",
    "roof",
    "sand",
    "sea",
    "shelf",
    "snow",
    "stairs",
    "tent",
    "towel",
    "wall-brick",
    "wall-stone",
    "wall-tile",
    "wall-wood",
    "water",
    "window-blind",
    "window",
    "tree",
    "fence",
    "ceiling",
    "sky",
    "cabinet",
    "table",
    "floor",
    "pavement",
    "mountain",
    "grass",
    "dirt",
    "paper",
    "food",
    "building",
    "rock",
    "wall",
    "rug",
]


class COCO_Model(DNN):
    def __init__(self, name):

        self.name = name

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(name))
        self.cfg.DOWNLOAD_CACHE = "/data2/kuntai/torch/detectron2/"

        # # to make it run on cpu, just for measurement purpose.
        # self.cfg.MODEL.DEVICE='cpu'
        # filter out those regions that has confidence score < 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)

        # reduce the examine script runtime
        self.predictor = None

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.class_ids = [0, 1, 2, 3, 5, 6, 7]

        if "Detection" in self.name:
            self.type = "Detection"
        elif "Keypoint" in self.name:
            self.type = "Keypoint"
        else:
            raise NotImplementedError

        self.keys = [
            "scores",
            "pred_classes",
            "pred_boxes",
            "pred_masks",
            "pred_keypoints",
            "proposals",
        ]

    # def cpu(self):

    #     self.predictor.model.cpu()
    #     self.logger.info(f"Place %s on CPU.", self.name)

    # def cuda(self):

    #     self.model.cuda()
    #     self.logger.info(f"Place %s on GPU.", self.name)

    def preprocess_image(self, image):

        assert (
            len(image.shape) == 4 and image.shape[0] == 1
        ), "Only deal with one image to avoid GPU memory overflow."

        h, w = image.shape[2], image.shape[3]

        if self.cfg.INPUT.FORMAT == "BGR":
            image = image[:, [2, 1, 0], :, :]
        else:
            assert self.cfg.INPUT.FORMAT == "RGB"

        image = image * 255
        transform = self.predictor.aug.get_transform(image[0].permute(1, 2, 0))
        image = F.interpolate(image, (transform.new_h, transform.new_w))

        return image, h, w, transform

    def inference(self, image, detach=False, grad=False):

        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)

        self.predictor.model.eval()

        image, h, w, _ = self.preprocess_image(image)

        with torch.enable_grad() if grad else torch.no_grad():
            ret = self.predictor.model(
                [{"image": image[0], "height": h, "width": w}]
            )[0]

        if detach:
            for key in ret:
                # this will also store the region proposal info
                ret[key] = ret[key].to("cpu")
        return ret

    def region_proposal(self, image, detach=False, grad=False):

        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)

        self.predictor.model.eval()

        image, h, w, _ = self.preprocess_image(image)

        with torch.enable_grad() if grad else torch.no_grad():
            model = self.predictor.model
            x = [{"image": image[0], "height": h, "width": w}]
            images = model.preprocess_image(x)
            features = model.backbone(images.tensor)
            proposals, logits = model.proposal_generator(images, features, None)

        ret = proposals[0]

        if detach:
            ret = ret.to("cpu")
        return ret

    # def calc_loss(self, image, result, args):

    #     if self.predictor is None:
    #         self.predictor = DefaultPredictor(self.cfg)

    #     self.predictor.model.train()

    #     result = self.filter_result(result, args)

    #     if result["instances"].has("pred_classes"):
    #         result["instances"].gt_classes = result["instances"].pred_classes
    #     if result["instances"].has("pred_boxes"):
    #         result["instances"].gt_boxes = result["instances"].pred_boxes
    #     if result["instances"].has("pred_masks"):
    #         result["instances"].gt_masks = result["instances"].pred_masks
    #     if result["instances"].has("pred_keypoints"):
    #         result["instances"].gt_keypoints = Keypoints(
    #             result["instances"].pred_keypoints
    #         )

    #     # convert result to target
    #     image, h, w, _ = self.preprocess_image(image)

    #     # Detectron2 models must be wrapped with EventStorage to run normally
    #     with EventStorage() as storage:

    #         ret = self.predictor.model(
    #             [
    #                 {
    #                     "image": image[0],
    #                     "height": h,
    #                     "width": w,
    #                     "instances": result["instances"],
    #                 }
    #             ]
    #         )

    #     return sum(ret.values())

    # def calc_dist(self, x, gt, args):

    #     if "Detection" in self.name:
    #         return self.calc_dist_detection(x, gt, args)
    #     else:
    #         assert False

    # def calc_dist_detection(self, x, gt, args):

    #     # get object of interest only
    #     x = self.filter_result(x, args, gt=False, confidence_check=False)
    #     gt = self.filter_result(gt, args, gt=True, confidence_check=True)

    #     x = x["instances"]
    #     gt = gt["instances"]

    #     assert len(x) > 0 and len(gt) > 0

    #     IoU = pairwise_iou(x.pred_boxes, gt.pred_boxes)

    #     loss_reg = torch.tensor([0.0]).cuda()

    #     for i in range(len(x)):

    #         val = IoU[i, :].max()

    #         p = x[i].scores

    #         if val < args.iou_threshold:
    #             # false positive encountered. Ignore.
    #             continue
    #         else:
    #             # true positive or false negative encountered. cover.
    #             loss_reg = loss_reg - (1 - p).log()

    #     return loss_reg

    def aggregate_inference_results(self, results, args):

        if "Detection" in self.name:
            return self.aggregate_inference_results_detection(results, args)
        else:
            raise NotImplementedError

    def aggregate_inference_results_detection(self, results, args):

        base = results[0]["instances"]

        scores = [base.scores]

        for result in results[1:]:

            result = copy.deepcopy(result["instances"])

            if len(base) == 0 or len(result) == 0:
                continue

            IoU = pairwise_iou(result.pred_boxes, base.pred_boxes)

            for i in range(len(result)):
                for j in range(len(base)):
                    if result.pred_classes[i] != base.pred_classes[j]:
                        IoU[i, j] = 0

            val, idx = IoU.max(dim=0)

            # clear those scores where IoU is way too small
            result[idx].scores[val < args.iou_threshold] = 0.0
            scores.append(result[idx].scores)

        scores = torch.cat([i.unsqueeze(0) for i in scores], dim=0)

        base.pred_scores = torch.tensor(scores).mean(dim=0)
        base.pred_std = torch.tensor(scores).std(dim=0)

        print(base.pred_std)

        return {"instances": base}

