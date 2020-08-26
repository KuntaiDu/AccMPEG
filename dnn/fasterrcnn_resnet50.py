

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import logging
from .dnn import DNN
from utils.bbox_utils import *

class FasterRCNN_ResNet50_FPN(DNN):

    def __init__(self):

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.logger = logging.getLogger(self.name())
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.is_cuda = False

    def cpu(self):

        self.model.cpu()
        self.is_cuda = False
        self.logger.info(f'Place {self.name()} on CPU.')

    def cuda(self):

        self.model.cuda()
        self.is_cuda = True
        self.logger.info(f'Place {self.name()} on GPU.')

    def inference(self, video, requires_grad = False):

        assert len(video.shape) == 4, 'The video tensor should be 4D'

        assert  self.is_cuda and video.is_cuda, 'The video tensor and the model must be placed on GPU to perform inference'

        context = None
        if requires_grad:
            context = torch.enable_grad
            assert video.requires_grad, 'Inference with gradient but the video input do not accept gradient'
            self.logger.info(f'Run inference on shape {video.shape} with gradient')
        else:
            context = torch.no_grad
            self.logger.info(f'Run inference on shape {video.shape} without gradient')

        with context():
            return self.model(video)

    def calc_accuracy(self, video, gt, args):
        '''
            Calculate the accuracy between video and gt using thresholds from args
        '''

        assert video.shape == gt.shape, f'The shape of video({video.shape}) and gt({gt.shape}) must be the same in order to calculate the accuracy'

        video_results, gt_results = self.inference(torch.cat([video, gt]))

        video_scores = video_results['scores']
        video_ind = video_scores > args.confidence_threshold
        video_bboxes = video_results['boxes'][video_ind, :]
        video_labels = video_results['labels'][video_ind]

        gt_scores = gt_results['scores']
        gt_ind = gt_scores > args.confidence_threshold
        gt_bboxes = gt_results['boxes'][gt_ind, :]
        gt_labels = gt_results['labels'][gt_ind]

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

        self.logger.info(f'Get an f1 score {f1}')

        return f1

        

