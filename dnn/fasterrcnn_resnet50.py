import logging
from pdb import set_trace

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utilities.bbox_utils import *

from .dnn import DNN

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
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
    "N/A",
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
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
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
    "N/A",
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
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
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
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class FasterRCNN_ResNet50_FPN(DNN):
    def __init__(self):

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.name = "FasterRCNN_ResNet50_FPN"

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.class_ids = [1, 2, 3, 4, 6, 7, 8]

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

        self.model.eval()

        video = [v for v in video]
        video = [F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in video]

        if nograd:
            with torch.no_grad():
                results = self.model(video)
        else:
            with torch.enable_grad():
                results = self.model(video)

        if detach:
            # detach and put everything to CPU.
            for result in results:
                for key in result:
                    result[key] = result[key].cpu().detach()

        return results

    def get_relevant_ind(self, labels):

        # filter out background classes
        relevant_labels = labels < 0
        for i in self.class_ids:
            relevant_labels = torch.logical_or(relevant_labels, labels == i)
        return relevant_labels

    def step(self, tensor):
        tensor = F.leaky_relu(100 * tensor, negative_slope=0.1)
        tensor = torch.min(tensor, torch.ones_like(tensor))
        tensor = torch.max(tensor, -0.05 * torch.ones_like(tensor))
        return tensor

    # def step(self, tensor):
    #     return (10 * tensor).sigmoid()

    def step2(self, tensor):
        return torch.where(
            tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor)
        )

    def filter_large_bbox(self, bboxes):

        size = (
            (bboxes[:, 2] - bboxes[:, 0]) / 1280 * (bboxes[:, 3] - bboxes[:, 1]) / 720
        )
        return size < 0.08

    def filter_results(
        self, video_results, confidence_threshold, cuda=False, train=False
    ):

        video_scores = video_results["scores"]
        video_ind = video_scores > confidence_threshold
        if not train:
            video_ind = torch.logical_and(
                video_ind, self.get_relevant_ind(video_results["labels"])
            )
            video_ind = torch.logical_and(
                video_ind, self.filter_large_bbox(video_results["boxes"])
            )
        video_scores = video_scores[video_ind]
        video_bboxes = video_results["boxes"][video_ind, :]
        video_labels = video_results["labels"][video_ind]
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
                video_ind.cpu(),
                video_scores.cpu(),
                video_bboxes.cpu(),
                video_labels.cpu(),
            )

    def calc_accuracy(self, video, gt, args):
        """
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        """

        assert video.keys() == gt.keys()

        f1s = []
        prs = []
        res = []
        tps = [torch.tensor(0)]
        fps = [torch.tensor(0)]
        fns = [torch.tensor(0)]

        for fid in video.keys():

            video_ind, video_scores, video_bboxes, video_labels = self.filter_results(
                video[fid], args.confidence_threshold
            )
            gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(
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

    def calc_loss(self, videos, gt_results, args, train=False):
        """
            Inference and calculate the loss between video and gt using thresholds from args
        """

        videos = [v for v in videos]
        videos = [F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in videos]

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

    # def calc_diff_acc(self, video, gt_results, args):
    #     '''
    #         Inference and calculate the loss between video and gt using thresholds from args
    #     '''

    #     assert len(video.shape) == 4, f'The shape of video({video.shape}) must be 4D.'

    #     # load the cached results to cuda
    #     gt_scores = gt_results['scores'].cuda()
    #     gt_ind = gt_scores > args.confidence_threshold
    #     gt_ind = torch.logical_and(gt_ind, self.get_relevant_ind(gt_results['labels'].cuda()))
    #     gt_ind = torch.logical_and(gt_ind, self.filter_large_bbox(gt_results['boxes'].cuda()))
    #     gt_bboxes = gt_results['boxes'][gt_ind, :].cuda()
    #     gt_labels = gt_results['labels'][gt_ind].cuda()

    #     # switch to eval mode
    #     if self.model.training:
    #         self.model.eval()

    #     with torch.enable_grad():
    #         video_results = self.model(video)[0]

    #     video_scores = video_results['scores']
    #     video_ind = (video_scores >= 0)
    #     video_ind = torch.logical_and(video_ind, self.get_relevant_ind(video_results['labels']))
    #     video_ind = torch.logical_and(video_ind, self.filter_large_bbox(video_results['boxes']))
    #     video_scores = video_scores[video_ind]
    #     video_bboxes = video_results['boxes'][video_ind, :]
    #     video_labels = video_results['labels'][video_ind]

    #     IoU = jaccard(video_bboxes, gt_bboxes)

    #     # let IoU = 0 if the label is wrong
    #     fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
    #     fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
    #     IoU[fat_video_labels != fat_gt_labels] = 0

    #     # enumerate all the labels
    #     tp = 0
    #     for gt_obj_id in range(len(gt_labels)):
    #         tp = tp + torch.min(self.step(IoU[:, gt_obj_id] - args.iou_threshold), self.step(video_scores - args.confidence_threshold)).max()

    #     # import pdb; pdb.set_trace()

    #     fp = torch.sum(self.step(video_scores - args.confidence_threshold)) - tp
    #     fn = len(gt_labels) - tp
    #     f1 = 2 * tp / (2 * tp + fp + fn)
    #     return f1, video_results

    def plot_results_on(self, gt, image, c, args, boxes=None, train=False):
        if gt == None:
            return image

        from PIL import Image, ImageColor, ImageDraw

        draw = ImageDraw.Draw(image)
        # load the cached results to cuda
        gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(
            gt, args.confidence_threshold, train=train
        )

        if boxes is None:
            for idx, box in enumerate(gt_bboxes):
                draw.rectangle(box.cpu().detach().tolist(), width=2, outline=c)
                draw.text(
                    box.cpu().tolist()[:2],
                    f"{gt_labels[idx].item()},{int(100 * gt_scores[idx])}",
                    fill="red",
                )
        else:
            rgb = ImageColor.getrgb(c)
            rgb_dim = (rgb[0] // 2, rgb[1] // 2, rgb[2] // 2)
            c_dim = "rgb(%d,%d,%d)" % rgb_dim
            IoU = jaccard(gt_bboxes, boxes)
            for idx, box in enumerate(gt_bboxes):
                if IoU[idx, :].sum() > args.iou_threshold:
                    draw.rectangle(box.cpu().tolist(), width=2, outline=c_dim)
                    draw.text(
                        box.cpu().tolist()[:2], f"{gt_labels[idx].item()}", fill="red"
                    )
                else:
                    draw.rectangle(box.cpu().tolist(), width=8, outline=c)
                    draw.text(
                        box.cpu().tolist()[:2], f"{gt_labels[idx].item()}", fill="red"
                    )

        return image

    def get_undetected_ground_truth_index(self, gt, video, args):

        video_ind, video_scores, video_bboxes, video_labels = self.filter_results(
            video, args.confidence_threshold
        )
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
        video = [F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in video]

        images, targets = self.model.transform(video, None)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            from collections import OrderedDict

            features = OrderedDict([("0", features)])
        proposals, _ = self.model.rpn(images, features, targets)

        return proposals
