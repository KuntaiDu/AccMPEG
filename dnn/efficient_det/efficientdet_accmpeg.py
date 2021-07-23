import logging
from pdb import set_trace
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils.bbox_utils import *

from torch.backends import cudnn
from dnn.efficient_det.backbone import EfficientDetBackbone
from dnn.efficient_det.efficientdet.utils import BBoxTransform, ClipBoxes
from dnn.efficient_det.utils.utils import preprocess, preprocess_accmpeg, invert_affine, postprocess

from dnn.dnn import DNN

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
threshold = 0.2
iou_threshold = 0.2

class EfficientDet(DNN):
    def __init__(self):

        self.model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
        self.model.load_state_dict(torch.load(f'dnn/efficient_det/weights/efficientdet-d{compound_coef}.pth'))
        self.model.requires_grad_(False)   
        self.model.eval()
        if use_cuda:  
            self.model = self.model.cuda()

        self.name = "EfficientDet"

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.class_ids = [0, 1, 2, 3, 4, 6, 7]

        self.is_cuda = False

    def inference(self, image, detach=False):

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        # image_path = "videos/dance_1_first_100_qp_24.mp4.pngs/0000000000.png"
        ori_imgs, framed_imgs, framed_metas = preprocess_accmpeg(image, max_size=1280)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

        # result
        out = invert_affine(framed_metas, out)
        # ret = out[0]

        # import pdb; pdb.set_trace()
        # if detach:
        #     for key in ret:
        #         # this will also store the region proposal info
        #         ret[key] = ret[key].to("cpu")
        return out

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

    def filter_results(self, video_results, confidence_threshold, cuda=False, train=False):
        # import pdb; pdb.set_trace()
        video_results = video_results[0]
        video_scores = video_results["scores"]
        video_ind = video_scores > confidence_threshold
        if not train:
            video_ind = np.logical_and(
                video_ind, self.get_relevant_ind(video_results["class_ids"])
            )
            video_ind = np.logical_and(
                video_ind, self.filter_large_bbox(video_results["rois"])
            )
        video_scores = video_scores[video_ind]
        video_bboxes = video_results["rois"][video_ind, :]
        video_labels = video_results["class_ids"][video_ind]
        video_ind = video_ind[video_ind]
        if cuda:
            return (
                video_ind.cuda(),
                video_scores.cuda(),
                video_bboxes.cuda(),
                video_labels.cuda(),
            )
        else:
            return (
                video_ind,
                video_scores,
                video_bboxes,
                video_labels,
            )

    def simple_filter(self, video_results, thres):
        video_results = video_results[0]
        video_scores = video_results["scores"]
        video_ind = video_scores > thres
        video_scores = video_scores[video_ind]
        video_bboxes = video_results["rois"][video_ind, :]
        video_labels = video_results["class_ids"][video_ind]
        video_ind = video_ind[video_ind]
        return (
            torch.tensor(video_ind),
            torch.tensor(video_scores),
            torch.tensor(video_bboxes),
            torch.tensor(video_labels),
        ) 

    def calc_accuracy(self, video, gt, args):
        """
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        """
        # import pdb; pdb.set_trace()

        assert video.keys() == gt.keys()

        f1s = []
        prs = []
        res = []
        tps = [torch.tensor(0)]
        fps = [torch.tensor(0)]
        fns = [torch.tensor(0)]

        for fid in video.keys():

            # video_ind, video_scores, video_bboxes, video_labels = self.filter_results(
            #     video[fid], args.confidence_threshold
            # )
            # gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(
            #     gt[fid], args.gt_confidence_threshold
            # )

            _, video_scores, video_bboxes, video_labels = self.simple_filter(
                video[fid], args.confidence_threshold
            )
            _, gt_scores, gt_bboxes, gt_labels = self.simple_filter(
                gt[fid], args.gt_confidence_threshold
            )

            if len(video_labels) == 0 or len(gt_labels) == 0:
                if len(video_labels) == 0 and len(gt_labels) == 0:
                    f1s.append(1.0)
                    prs.append(1.0)
                    res.append(1.0)
                else:
                    f1s.append(0.0)
                    prs.append(0.0)
                    res.append(0.0)
                continue
            
            IoU = jaccard(video_bboxes, gt_bboxes)

            # let IoU = 0 if the label is wrong
            fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
            fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
            IoU[fat_video_labels != fat_gt_labels] = 0

            # calculate f1
            tp = 0

            for i in range(len(gt_labels)):
                tp = (
                    tp
                    + torch.min(
                        self.step2(IoU[:, i] - args.iou_threshold),
                        self.step2(video_scores - args.confidence_threshold),
                    ).max()
                )
            tp = min(
                [
                    tp,
                    len(gt_labels),
                    len(video_labels[video_scores > args.confidence_threshold]),
                ]
            )
            fn = len(gt_labels) - tp
            fp = len(video_labels[video_scores > args.confidence_threshold]) - tp

            f1 = 2 * tp / (2 * tp + fp + fn)
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            # import pdb; pdb.set_trace()

            f1s.append(f1)
            prs.append(pr)
            res.append(re)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

            # if fid % 10 == 9:
            #     #pass
            #     print('f1:', torch.tensor(f1s[-9:]).mean().item())
            #     print('pr:', torch.tensor(prs[-9:]).mean().item())
            #     print('re:', torch.tensor(res[-9:]).mean().item())

        return {
            "f1": torch.tensor(f1s).mean().item(),
            "pr": torch.tensor(prs).mean().item(),
            "re": torch.tensor(res).mean().item(),
            "tp": sum(tps).item(),
            "fp": sum(fps).item(),
            "fn": sum(fns).item(),
        }