import logging
from pdb import set_trace

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from utilities.bbox_utils import *
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

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


class Detr_ResNet101(DNN):
    def __init__(self):

        self.name = "detr_resnet101"
        self.source = "facebookresearch/detr"
        self.segm_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True)
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        self.segm_thresh = 0.7
        self.model.eval()
        self.segm_model.eval()


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

        video_scores, labels = torch.max(torch.nn.functional.softmax(results[0]['pred_logits'], dim=2)[0], dim=1)
        boxes = results[0]["pred_boxes"].squeeze(0)
        _,h,w = video[0].shape
        boxes = boxes*torch.tensor([w,h,w,h])
        boxes = [[box[0] - 0.5*box[2], box[1] - 0.5*box[3], box[0] + 0.5*box[2], box[1] + 0.5*box[3]] for box in boxes]
        #for box in boxes:
        #    b = [box[0] - 0.5*box[2], box[1] - 0.5*box[3], box[0] + 0.5*box[2], box[1] + 0.5*box[3]]
        #    bx.append(b)



        ret = Instances(
            image_size=(h, w),
            pred_boxes=Boxes(boxes),
            #pred_boxes=Boxes(bx),
            scores=video_scores,
            pred_classes=labels
        )

        if detach:
            ret = ret.to("cpu")

        return {"instances": ret}

        #return results

    def filter_result(self, result, args, gt=False):
    
        scores = result["instances"].scores
        class_ids = result["instances"].pred_classes

        inds = scores < 0
        for i in self.class_ids:
            inds = inds | (class_ids == i)
        if gt:
            inds = inds & (scores > args.gt_confidence_threshold)
        else:
            inds = inds & (scores > args.confidence_threshold)

        result["instances"] = result["instances"][inds]

        return result

    #def filter_result(
    #    self, video_results, confidence_threshold, cuda=False, train=False
    #):

    #    if not isinstance(video_results, dict):
    #        video_results=video_results[0]
    #    video_scores, labels = torch.max(torch.nn.functional.softmax(video_results['pred_logits']), dim=2)
    #    video_scores = video_scores[0]
    #    labels = labels[0]
    #    boxes = video_results["pred_boxes"].squeeze(0)

    #    video_ind = video_scores > confidence_threshold
    #    if not train:
    #        video_ind = torch.logical_and(
    #            video_ind, self.get_relevant_ind(labels)
    #        )
    #        video_ind = torch.logical_and(
    #            video_ind, self.filter_large_bbox(boxes)
    #        )
    #    video_scores = video_scores[video_ind]
    #    video_bboxes = boxes[video_ind, :]
    #    video_labels = labels[video_ind]
    #    video_ind = video_ind[video_ind]
    #    if cuda:
    #        return (
    #            video_ind.cuda(),
    #            video_scores.cuda(),
    #            video_bboxes.cuda(),
    #            video_labels.cuda(),
    #        )
    #    else:
    #        return (
    #            video_ind.cpu(),
    #            video_scores.cpu(),
    #            video_bboxes.cpu(),
    #            video_labels.cpu(),
    #        )

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

            if (2 * tp + fp + fn) == 0:
                f1 = 1.0
            else:
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
            #"sum_f1": (2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn))
            #"f1s": f1s,
            #"prs": prs,
            #"res": res,
            #"tps": tps,
            #"fns": fns,
            #"fps": fps,
        }
    #def calc_accuracy(self, video, gt, args):
    #    """
    #        Calculate the accuracy between video and gt using thresholds from args based on inference results
    #    """

    #    assert video.keys() == gt.keys()

    #    f1s = []
    #    prs = []
    #    res = []
    #    tps = [torch.tensor(0)]
    #    fps = [torch.tensor(0)]
    #    fns = [torch.tensor(0)]

    #    for fid in video.keys():

    #        video_ind, video_scores, video_bboxes, video_labels = self.filter_result(
    #            video[fid], args.confidence_threshold
    #        )
    #        gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_result(
    #            gt[fid], args.gt_confidence_threshold
    #        )
    #        if len(video_labels) == 0 or len(gt_labels) == 0:
    #            if len(video_labels) == 0 and len(gt_labels) == 0:
    #                f1s.append(1.0)
    #                prs.append(1.0)
    #                res.append(1.0)
    #            else:
    #                f1s.append(0.0)
    #                prs.append(0.0)
    #                res.append(0.0)
    #            continue

    #        IoU = jaccard(video_bboxes, gt_bboxes)

    #        # let IoU = 0 if the label is wrong
    #        fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
    #        fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
    #        IoU[fat_video_labels != fat_gt_labels] = 0

    #        # calculate f1
    #        tp = 0

    #        for i in range(len(gt_labels)):
    #            tp = (
    #                tp
    #                + torch.min(
    #                    self.step2(IoU[:, i] - args.iou_threshold),
    #                    self.step2(video_scores - args.confidence_threshold),
    #                ).max()
    #            )
    #        tp = min(
    #            [
    #                tp,
    #                len(gt_labels),
    #                len(video_labels[video_scores > args.confidence_threshold]),
    #            ]
    #        )
    #        fn = len(gt_labels) - tp
    #        fp = len(video_labels[video_scores > args.confidence_threshold]) - tp

    #        f1 = 2 * tp / (2 * tp + fp + fn)
    #        pr = tp / (tp + fp)
    #        re = tp / (tp + fn)
    #        # import pdb; pdb.set_trace()

    #        f1s.append(f1)
    #        prs.append(pr)
    #        res.append(re)
    #        tps.append(tp)
    #        fps.append(fp)
    #        fns.append(fn)

    #        # if fid % 10 == 9:
    #        #     #pass
    #        #     print('f1:', torch.tensor(f1s[-9:]).mean().item())
    #        #     print('pr:', torch.tensor(prs[-9:]).mean().item())
    #        #     print('re:', torch.tensor(res[-9:]).mean().item())

    #    return {
    #        "f1": torch.tensor(f1s).mean().item(),
    #        "pr": torch.tensor(prs).mean().item(),
    #        "re": torch.tensor(res).mean().item(),
    #        "tp": sum(tps).item(),
    #        "fp": sum(fps).item(),
    #        "fn": sum(fns).item(),
    #    }

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

    def calc_loss(self, videos, gt_results, args, train=False):
        """
            Inference and calculate the loss between video and gt using thresholds from args
        """

        if self.is_cuda():
            videos = [v.cuda() for v in videos]
        else:
            videos = [v for v in videos]
        videos = [F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in videos]

        def transform_result(gt_result):
            # calculate the ground truth
            gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_result(
                gt_result, args.confidence_threshold, cuda=True, train=train
            )
            # construct targets
            target = {"boxes": gt_bboxes, "labels": gt_labels}
            return target

        targets = [transform_result(gt_result) for gt_result in gt_results]

        # we use model in eval mode because loss claculation is handled by a separate criterion module
        self.model.eval()
        #assert self.is_cuda, "Model must be placed on GPU"
        with torch.no_grad():
            model_output = self.model(videos)

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        num_classes = 91
        eos_coef = 0.1
        bbox_loss_coef = 5
        losses = ['boxes','cardinality']
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=eos_coef, losses=losses)
        num_boxes = sum(len(t["labels"]) for t in targets)
        indices = matcher(model_output, targets)
        loss_vals = {}
        for loss in losses:
            lossval = criterion.get_loss(loss, model_output, targets, indices, num_boxes)
            loss_vals.update(lossval)

        return sum(loss_vals.values())

    def region_proposal(self, video):
        self.segm_model.eval()

        video = [v for v in video]
        video = [F.interpolate(v[None, :, :, :], size=(720, 1280))[0] for v in video]

        masks = self.segm_model(video)["pred_masks"]
        masks = F.softmax(masks)
        binary_masks = (masks > self.segm_thresh).type(torch.uint8)
        composite_mask = (binary_masks.sum(dim=1) > 0)

        return composite_mask
