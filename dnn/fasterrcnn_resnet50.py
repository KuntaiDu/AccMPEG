

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import logging
from .dnn import DNN
from utils.bbox_utils import *

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class FasterRCNN_ResNet50_FPN(DNN):

    def __init__(self):

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.class_ids = [3, 6, 7, 8]

        self.is_cuda = False

    def cpu(self):

        self.model.cpu()
        self.is_cuda = False
        self.logger.info(f'Place {self.name} on CPU.')

    def cuda(self):

        self.model.cuda()
        self.is_cuda = True
        self.logger.info(f'Place {self.name} on GPU.')

    def inference(self, video, detach=False):
        '''
            Generate raw inference results
        '''

        assert len(video.shape) == 4, 'The video tensor should be 4D'

        assert  self.is_cuda and video.is_cuda, 'The video tensor and the model must be placed on GPU to perform inference'

        with torch.no_grad():
            results = self.model(video)

        if detach:
            # detach and put everything to CPU.
            for result in results:
                for key in result:
                    result[key] = result[key].cpu().detach()

        return results

    def get_relevant_ind(self, labels):

        inds = (labels < 0)

        for class_id in self.class_ids:
            inds = torch.logical_or(inds, labels == class_id)

        return inds
        
                

    def calc_accuracy(self, video, gt, args):
        '''
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        '''

        assert video.keys() == gt.keys()

        accuracies = []

        for fid in video.keys():
            
            video_scores = video[fid]['scores']
            video_ind = video_scores > args.confidence_threshold
            video_ind = torch.logical_and(video_ind, self.get_relevant_ind(video[fid]['labels']))
            video_bboxes = video[fid]['boxes'][video_ind, :]
            video_labels = video[fid]['labels'][video_ind]

            gt_scores = gt[fid]['scores']
            gt_ind = gt_scores > args.confidence_threshold
            gt_ind = torch.logical_and(gt_ind, self.get_relevant_ind(gt[fid]['labels']))
            gt_bboxes = gt[fid]['boxes'][gt_ind, :]
            gt_labels = gt[fid]['labels'][gt_ind]

            IoU = jaccard(video_bboxes, gt_bboxes)

            # let IoU = 0 if the label is wrong
            fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
            fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
            IoU[fat_video_labels != fat_gt_labels] = 0

            # calculate f1
            tp, fp, fn = 0, 0, 0

            for i in range(len(gt_labels)):
                if (IoU[:, i] > args.iou_threshold).sum() > 0:
                    tp += 1
                else:
                    fn += 1
            fp = len(video_labels) - tp

            f1 = None
            if fp + fn == 0:
                f1 = 1
            else:
                f1 = 2 * tp / (2 * tp + fp + fn)

            self.logger.info(f'Get an f1 score {f1} at frame {fid}')

            accuracies.append(f1)

        return torch.tensor(accuracies).mean()

    def calc_loss(self, video, gt, args):
        '''
            Inference and calculate the loss between video and gt using thresholds from args
        '''

        assert video.shape == gt.shape, f'The shape of video({video.shape}) and gt({gt.shape}) must be the same in order to calculate the loss'
        assert len(video.shape) == 4, f'The shape of video({video.shape}) must be 4D.'

        # inference, and obtain the inference results
        self.model.eval()
        gt_results = self.inference(gt)[0]
        gt_scores = gt_results['scores']
        gt_ind = gt_scores > args.confidence_threshold
        gt_ind = torch.logical_and(gt_ind, self.get_relevant_ind(gt_results['labels']))
        gt_bboxes = gt_results['boxes'][gt_ind, :]
        gt_labels = gt_results['labels'][gt_ind]

        # construct targets
        targets = [{
            'boxes': gt_bboxes,
            'labels': gt_labels
        }]

        # switch the model to training mode to obtain loss
        self.model.train()
        self.model.zero_grad()
        assert self.is_cuda and video.is_cuda, 'The video tensor and the model must be placed on GPU to perform inference'
        with torch.enable_grad():
            losses = self.model(video, targets)

        return sum(loss for loss in losses.values())
