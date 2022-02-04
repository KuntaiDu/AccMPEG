"""
    Implemented based on https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
"""


import logging
import os
from pathlib import Path
from pdb import set_trace
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wget
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from dnn.dnn import DNN
from dnn.efficient_det.backbone import EfficientDetBackbone
from dnn.efficient_det.efficientdet.utils import BBoxTransform, ClipBoxes
from PIL import Image
from torch.backends import cudnn
from torchvision.ops.boxes import batched_nms
from utilities.bbox_utils import *

obj_list = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "",
    "backpack",
    "umbrella",
    "",
    "",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "",
    "dining table",
    "",
    "",
    "toilet",
    "",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

compound_coef = 0
use_cuda = True
use_float16 = False


class EfficientDet(DNN):
    def __init__(self):

        self.name = "EfficientDet"
        self.logger = logging.getLogger(self.name)

        self.model = EfficientDetBackbone(
            compound_coef=compound_coef, num_classes=len(obj_list)
        )

        self.model_pth = Path(
            f"dnn/efficient_det/weights/efficientdet-d{compound_coef}.pth"
        )
        if not self.model_pth.exists():
            self.logger.warning(
                "Pretrained EfficientDet model not found. Downloading..."
            )
            self.model_pth.parent.mkdir(exist_ok=True, parents=True)
            wget.download(
                f"https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d{compound_coef}.pth",
                out=str(self.model_pth),
            )
        self.model.load_state_dict(torch.load(self.model_pth))
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.cuda()

        # class ids: all vehicles and persons except for train.
        self.class_ids = [0, 1, 2, 3, 4, 6]
        # code refactor version
        # self.model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
        # self.model.load_state_dict(torch.load(f'dnn/efficient_det/weights/efficientdet-d{compound_coef}.pth'))
        # self.model.requires_grad_(False)
        # self.model.eval()
        # self.model.cuda()
        # self.name = "EfficientDet"

        # self.logger = logging.getLogger(self.name)
        # handler = logging.NullHandler()
        # self.logger.addHandler(handler)

        # self.class_ids = [0, 1, 2, 3, 4, 6, 7]

        self.coco_normalize = T.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        self.type = "Detection"

    def inference(self, image, detach=False, grad=False):

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        _, _, h, w = image.shape

        image = [self.coco_normalize(img) for img in image]

        ori_imgs, x, framed_metas = preprocess_accmpeg(image, max_size=1280)
        x = torch.stack(x)

        if not x.is_cuda:
            x = x.cuda()

        # model predict
        features, regression, classification, anchors = self.model(x)

        # postprocessing
        threshold = 0.2
        iou_threshold = 0.2
        out = postprocess(
            x,
            anchors,
            regression,
            classification,
            regressBoxes,
            clipBoxes,
            threshold,
            iou_threshold,
        )

        # result
        out = invert_affine(framed_metas, out)

        # construct COCO-style result
        ret = Instances(
            image_size=(h, w),
            pred_boxes=Boxes(out[0]["rois"]),
            scores=out[0]["scores"],
            pred_classes=out[0]["class_ids"],
        )

        if detach:
            ret = ret.to("cpu")

        return {"instances": ret}

    def get_relevant_ind(self, labels):

        # filter out background classes
        relevant_labels = labels < 0
        for i in self.class_ids:
            relevant_labels = np.logical_or(relevant_labels, labels == i)
        return relevant_labels

    def filter_large_bbox(self, bboxes):
        size = (
            (bboxes[:, 2] - bboxes[:, 0])
            / 1280
            * (bboxes[:, 3] - bboxes[:, 1])
            / 720
        )
        return size < 0.08

    def step2(self, tensor):
        return torch.where(
            tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor)
        )


def aspectaware_resize_padding(
    image, width, height, interpolation=None, means=None
):

    # 720, 1280, 3
    c, old_h, old_w = image.shape

    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    padding_h = height - new_h
    padding_w = width - new_w

    # pad original image
    image = F.pad(image, (0, padding_w, 0, padding_h))

    return (
        image,
        new_w,
        new_h,
        old_w,
        old_h,
        padding_w,
        padding_h,
    )


def preprocess_accmpeg(images, max_size=1280):
    imgs_meta = [
        aspectaware_resize_padding(image, 1280, 1280, means=None)
        for image in images
    ]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    return images, framed_imgs, framed_metas


def postprocess(
    x,
    anchors,
    regression,
    classification,
    regressBoxes,
    clipBoxes,
    threshold,
    iou_threshold,
):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )
            continue

        classification_per = classification[
            i, scores_over_thresh[i, :], ...
        ].permute(1, 0)
        transformed_anchors_per = transformed_anchors[
            i, scores_over_thresh[i, :], ...
        ]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(
            transformed_anchors_per,
            scores_per[:, 0],
            classes_,
            iou_threshold=iou_threshold,
        )

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append(
                {"rois": boxes_, "class_ids": classes_, "scores": scores_,}
            )
        else:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )

    return out


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]["rois"]) == 0:
            continue
        else:
            if metas is float:
                preds[i]["rois"][:, [0, 2]] = (
                    preds[i]["rois"][:, [0, 2]] / metas
                )
                preds[i]["rois"][:, [1, 3]] = (
                    preds[i]["rois"][:, [1, 3]] / metas
                )
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / (
                    new_w / old_w
                )
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / (
                    new_h / old_h
                )
    return preds
