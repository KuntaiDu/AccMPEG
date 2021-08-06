'''
    Implemented based on https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
'''



import logging
from pdb import set_trace
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from utils.bbox_utils import *

from torch.backends import cudnn
from dnn.efficient_det.backbone import EfficientDetBackbone
from dnn.efficient_det.efficientdet.utils import BBoxTransform, ClipBoxes

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from torchvision.ops.boxes import batched_nms
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from dnn.dnn import DNN
from typing import Union
from pathlib import Path
import wget
from PIL import Image



obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

compound_coef = 0
use_cuda = True
use_float16 = False


class EfficientDet(DNN):
    def __init__(self):

        self.name = "EfficientDet"
        self.logger = logging.getLogger(self.name)

        self.model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))

        self.model_pth = Path(f'dnn/efficient_det/weights/efficientdet-d{compound_coef}.pth')
        if not self.model_pth.exists():
            self.logger.warning("Pretrained EfficientDet model not found. Downloading...")
            self.model_pth.parent.mkdir(exist_ok=True, parents=True)
            wget.download(f"https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d{compound_coef}.pth", out=str(self.model_pth))
        self.model.load_state_dict(torch.load(self.model_pth))
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.cuda()
        

        self.class_ids = [0, 1, 2, 3, 4, 6, 7]

        self.coco_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    def inference(self, image, detach=False):

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
        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

        # result
        out = invert_affine(framed_metas, out)

        # construct COCO-style result
        ret = Instances(
            image_size=(h, w),
            pred_boxes=Boxes(out[0]['rois']),
            scores=out[0]['scores'],
            pred_classes=out[0]['class_ids']
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
            (bboxes[:, 2] - bboxes[:, 0]) / 1280 * (bboxes[:, 3] - bboxes[:, 1]) / 720
        )
        return size < 0.08

    def step2(self, tensor):
        return torch.where(
            tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor)
        )

    def filter_result(self, result, args, gt=False, confidence_filter=True):
    
        scores = result["instances"].scores
        class_ids = result["instances"].pred_classes

        inds = scores < 0
        for i in self.class_ids:
            inds = inds | (class_ids == i)

        if conconfidence_filter:
            if gt:
                inds = inds & (scores > args.gt_confidence_threshold)
            else:
                inds = inds & (scores > args.confidence_threshold)

        result["instances"] = result["instances"][inds]

        return result

    def calc_accuracy(self, result_dict, gt_dict, args):
    
        from detectron2.structures.boxes import pairwise_iou

        assert (
            result_dict.keys() == gt_dict.keys()
        ), "Result and ground truth must contain the same number of frames."

        f1s = []
        prs = []
        res = []
        tps = []
        fps = []
        fns = []

        for fid in result_dict.keys():
            result = result_dict[fid]
            gt = gt_dict[fid]

            result = self.filter_result(result, args, False)
            gt = self.filter_result(gt, args, True)

            result = result["instances"]
            gt = gt["instances"]

            if len(result) == 0 or len(gt) == 0:
                if len(result) == 0 and len(gt) == 0:
                    f1s.append(1.0)
                    prs.append(1.0)
                    res.append(1.0)
                else:
                    f1s.append(0.0)
                    prs.append(0.0)
                    res.append(0.0)

            IoU = pairwise_iou(result.pred_boxes, gt.pred_boxes)

            for i in range(len(result)):
                for j in range(len(gt)):
                    if result.pred_classes[i] != gt.pred_classes[j]:
                        IoU[i, j] = 0

            tp = 0

            for i in range(len(gt)):
                if sum(IoU[:, i] > args.iou_threshold):
                    tp += 1
            fn = len(gt) - tp
            fp = len(result) - tp
            fp = max(fp, 0)

            f1 = 2 * tp / (2 * tp + fp + fn)
            if tp + fp == 0:
                pr = 1.0
            else:
                pr = tp / (tp + fp)
            if tp + fn == 0:
                re = 1.0
            else:
                re = tp / (tp + fn)

            f1s.append(f1)
            prs.append(pr)
            res.append(re)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

        return {
            "f1": torch.tensor(f1s).mean().item(),
            "pr": torch.tensor(prs).mean().item(),
            "re": torch.tensor(res).mean().item(),
            "tp": torch.tensor(tps).sum().item(),
            "fp": torch.tensor(fps).sum().item(),
            "fn": torch.tensor(fns).sum().item(),
            "f1s": f1s,
            "prs": prs,
            "res": res,
            "tps": tps,
            "fns": fns,
            "fps": fps,
        }

    def visualize(self, image, result, args):
        # set_trace()
        result = self.filter_result(result, args, gt=False, confidence_filter=False)
        v = Visualizer(
            image, MetadataCatalog.get('coco2017_train'), scale=1
        )
        out = v.draw_instance_predictions(result["instances"])
        return Image.fromarray(out.get_image(), "RGB")



def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    
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

    return image, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess_accmpeg(images, max_size=1280):
    imgs_meta = [aspectaware_resize_padding(image, 1280, 1280, means=None) for image in images]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    return images, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_,
                'class_ids': classes_,
                'scores': scores_,
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out

def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds