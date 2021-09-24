import logging
from pdb import set_trace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
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

label_colors = torch.Tensor(
    [
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
)


class FCN_ResNet50(DNN):
    def __init__(self):

        self.model = fcn_resnet101(pretrained=True)
        self.model.eval()

        self.class_ids = [0, 2, 6, 7, 14, 15]

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.transform = T.Compose(
            [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.is_cuda = False

    def cpu(self):

        self.model.cpu()
        self.is_cuda = False
        self.logger.info(f"Place {self.name} on CPU.")

    def cuda(self):

        self.model.cuda()
        self.is_cuda = True
        self.logger.info(f"Place {self.name} on GPU.")

    def parallel(self, local_rank):
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[local_rank], find_unused_parameters=True
        )

    def inference(self, video, detach=False, nograd=True):
        """
            Generate inference results. Will put results on cpu if detach=True.
        """

        if not self.is_cuda:
            self.cuda()

        self.model.eval()

        # video = [v for v in video]
        # video = [F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in video]
        video = F.interpolate(video, size=(720, 1280))

        if nograd:
            with torch.no_grad():
                results = self.model(video)
        else:
            with torch.enable_grad():
                results = self.model(video)

        results = results["out"]

        results = results[:, self.class_ids, :, :]
        results = results.argmax(1).byte()

        if detach:
            results = results.detach().cpu()

        return results

    # def step(self, tensor):
    #     return (10 * tensor).sigmoid()

    def filter_results(
        self, video_results, confidence_threshold, cuda=False, train=False
    ):

        raise NotImplementedError()

    def calc_accuracy(self, video, gt, args):
        """
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        """

        assert video.keys() == gt.keys()

        accs = []

        for fid in video.keys():

            video_result = video[fid]
            gt_result = gt[fid]

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

        videos = [v for v in videos]
        videos = [
            F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in videos
        ]

        def transform_result(gt_result):
            # calculate the ground truth
            gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(
                gt_result, args.confidence_threshold, cuda=True, train=train
            )
            # construct targets
            target = {"boxes": gt_bboxes, "labels": gt_labels}
            return target

        targets = [transform_result(gt_result) for gt_result in gt_results]

        # switch the model to training mode to obtain loss
        self.model.train()
        self.model.zero_grad()
        assert self.is_cuda, "Model must be placed on GPU"
        losses = self.model(videos, targets)

        # return losses["loss_classifier"] + losses["loss_box_reg"]
        return sum(losses.values())

    def plot_results_on(self, gt, image, c, args, boxes=None, train=False):
        if gt == None:
            return image

        from PIL import Image, ImageColor, ImageDraw

        gt = gt[None, :, :]

        r, g, b = [torch.zeros_like(gt).float() for i in range(3)]

        for class_id in range(len(self.class_ids)):
            ind = gt == class_id
            r[ind] = label_colors[class_id, 0]
            g[ind] = label_colors[class_id, 1]
            b[ind] = label_colors[class_id, 2]

        label = torch.cat([r, g, b]) / 255.0
        label_image = T.ToPILImage()(label)

        image = image.convert("RGBA")
        label_image = label_image.convert("RGBA")
        return Image.blend(image, label_image, alpha=0.5)

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

    def region_proposal(self, video):
        self.model.eval()

        video = [v for v in video]
        video = [
            F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in video
        ]

        images, targets = self.model.transform(video, None)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            from collections import OrderedDict

            features = OrderedDict([("0", features)])
        proposals, _ = self.model.rpn(images, features, targets)

        return proposals
