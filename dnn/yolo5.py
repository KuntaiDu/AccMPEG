import logging
from pdb import set_trace

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from utils.bbox_utils import *

from .dnn import DNN


class Yolo5s(DNN):
    def __init__(self):

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()


        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.class_ids = [1, 2, 3, 4, 6, 7, 8]

        self.is_cuda = False

        self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.model.cuda()

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

        # perform COCO normalization
        video = [self.transform(v) for v in video]

        if nograd:
            with torch.no_grad():
                results = self.model(video)
        else:
            with torch.enable_grad():
                results = self.model(video)

        results = [results]
        if detach:
            # detach and put everything to CPU.
            for result in results:
                for key in result:
                    result[key] = result[key].cpu().detach()

        out = {}
        out['video_scores'] = results.xyxy[0][:,4]
        out['labels'] = results.xyxy[0][:,5]
        out['boxes'] = results.xyxy[0][:,:4]
        return out

    def filter_results(
        self, video_results, confidence_threshold, cuda=False, train=False
    ):

        video_scores = video_results["video_scores"]
        labels = video_results["labels"]
        boxes = video_results["boxes"]

        video_ind = video_scores > confidence_threshold
        if not train:
            video_ind = torch.logical_and(
                video_ind, self.get_relevant_ind(labels)
            )
            video_ind = torch.logical_and(
                video_ind, self.filter_large_bbox(boxes)
            )
        video_scores = video_scores[video_ind]
        video_bboxes = boxes[video_ind, :]
        video_labels = labels[video_ind]
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

    def get_relevant_ind(self, labels):

        # filter out background classes
        relevant_labels = labels < 0
        for i in self.class_ids:
            relevant_labels = torch.logical_or(relevant_labels, labels == i)
        return relevant_labels

    def filter_large_bbox(self, bboxes):

        size = (
            (bboxes[:, 2] - bboxes[:, 0]) / 1280 * (bboxes[:, 3] - bboxes[:, 1]) / 720
        )
        return size < 0.08
