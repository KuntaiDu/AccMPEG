import logging
from pdb import set_trace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    fcn_resnet50,
    fcn_resnet101,
)
from utilities.bbox_utils import *

from .dnn import DNN

COCO_NAMES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

label_colors = [
    (0, 0, 0),  # 0=background
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
]

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class Segmentation(DNN):
    def __init__(self, name):

        model_name = name.split("/")[-1]
        exec(f"self.model = {model_name}(pretrained=True)")
        self.model.eval()
        self.name = name

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.class_ids = [0, 2, 6, 7, 14, 15]

        self.transform = T.Compose(
            [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.metadata = MetadataCatalog.get("my_fcn_resnet101")
        self.metadata.stuff_classes = COCO_NAMES
        self.metadata.stuff_colors = label_colors

        self.is_cuda = False

    def cpu(self):

        self.model.cpu()
        self.is_cuda = False
        self.logger.info(f"Place {self.name} on CPU.")

    def cuda(self):

        self.model.cuda()
        self.is_cuda = True
        self.logger.info(f"Place {self.name} on GPU.")

    # def parallel(self, local_rank):
    #     self.model = torch.nn.parallel.DistributedDataParallel(
    #         self.model, device_ids=[local_rank], find_unused_parameters=True
    #     )

    def inference(self, video, detach=False):
        """
            Generate inference results. Will put results on cpu if detach=True.
        """

        self.model.eval()
        if not self.is_cuda:
            self.cuda()
        if not video.is_cuda:
            video = video.cuda()

        video = F.interpolate(video, size=(720, 1280))
        video = torch.cat([self.transform(v) for v in video.split(1)])

        with torch.no_grad():
            results = self.model(video)

        results = results["out"]

        """ newly added here"""
        results = results[:, self.class_ids, :, :]

        results = results.argmax(1).byte()

        if detach:
            results = results.detach().cpu()

        return results

    # def step(self, tensor):
    #     return (10 * tensor).sigmoid()

    def filter_result(self, video_results, args):

        """
        BYPASS THIS FUNCTION
        """

        return video_results

        from skimage import measure

        bin_video = torch.where(
            video_results > 0,
            torch.ones_like(video_results),
            torch.zeros_like(video_results),
        )

        # set_trace()

        bin_video = torch.tensor(
            measure.label(bin_video.numpy().astype(int)), dtype=torch.int32
        )
        nclass = torch.max(bin_video).item()
        mask = torch.zeros_like(video_results)

        for i in range(1, nclass + 1):
            size = torch.sum(bin_video == i) * 1.0 / bin_video.numel()
            if size < args.size_bound:
                mask[bin_video == i] = 1

        return video_results * mask

    def calc_accuracy(self, video, gt, args):
        """
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        """

        assert video.keys() == gt.keys()

        accs = []

        for fid in video.keys():

            if fid % 10 == 0:
                print(fid)

            video_result = self.filter_result(video[fid], args)
            gt_result = self.filter_result(gt[fid], args)

            mask = ~((video_result == 0) & (gt_result == 0))
            correct = (video_result == gt_result) & mask

            ncorrect = len(correct.nonzero(as_tuple=False))
            nall = len(mask.nonzero(as_tuple=False))
            if nall != 0:
                accs.append(ncorrect / nall)
            else:
                accs.append(1.0)

            # if fid % 10 == 9:
            #     #pass
            #     print('f1:', torch.tensor(f1s[-9:]).mean().item())
            #     print('pr:', torch.tensor(prs[-9:]).mean().item())
            #     print('re:', torch.tensor(res[-9:]).mean().item())

        return {"acc": torch.Tensor(accs).mean().item()}

    def calc_loss(self, videos, gt_results, args, train=False):
        """
            Inference and calculate the loss between video and gt using thresholds from args
        """

        if not self.is_cuda:
            self.cuda()
        if not videos.is_cuda:
            videos = videos.cuda()

        videos = F.interpolate(videos, size=(720, 1280))
        videos = torch.cat([self.transform(v) for v in videos.split(1)])

        targets = gt_results.cuda()

        # switch the model to training mode to obtain loss
        # set_trace()
        return FocalLoss(weight=torch.ones(len(COCO_NAMES)).cuda())(
            self.model(videos)["out"], targets[:, :, :].long()
        )

    def visualize(self, image, result, args):
        # set_trace()
        result = self.filter_result(result, args)
        v = Visualizer(image, self.metadata, scale=1)
        out = v.draw_sem_seg(result[0])
        return Image.fromarray(out.get_image(), "RGB")

    def get_undetected_ground_truth_index(self, gt, video, args):

        (
            video_ind,
            video_scores,
            video_bboxes,
            video_labels,
        ) = self.filter_results(video, args.confidence_threshold)
        gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(
            gt, args.confidence_threshold
        )

        # get IoU and clear the IoU of mislabeled objects
        IoU = jaccard(video_bboxes, gt_bboxes)
        fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
        fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
        IoU[fat_video_labels != fat_gt_labels] = 0

        return (IoU > args.iou_threshold).sum(dim=0) == 0
