

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import torch.nn.functional as F
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
        # self.class_ids = [3, 6, 7, 8]

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
            Generate inference results. Will put results on cpu if detach=True.
        '''

        assert len(video.shape) == 4, 'The video tensor should be 4D'

        assert  self.is_cuda and video.is_cuda, 'The video tensor and the model must be placed on GPU to perform inference'

        self.model.eval()

        with torch.no_grad():
            results = self.model(video)

        if detach:
            # detach and put everything to CPU.
            for result in results:
                for key in result:
                    result[key] = result[key].cpu().detach()

        return results

    def get_relevant_ind(self, labels):

        # filter out background classes
        return labels > 0

    def step(self, tensor):
        tensor = F.leaky_relu(100 * tensor, negative_slope=0.1)
        tensor = torch.min(tensor, torch.ones_like(tensor))
        tensor = torch.max(tensor, -0.05 * torch.ones_like(tensor))
        return tensor

    # def step(self, tensor):
    #     return (10 * tensor).sigmoid()
        
    def step2(self, tensor):
        return torch.where(tensor > 0, torch.ones_like(tensor), torch.zeros_like(tensor))

    def filter_large_bbox(self, bboxes):

        size = (bboxes[:, 2] - bboxes[:, 0]) / 1280 * (bboxes[:, 3] - bboxes[:, 1]) / 720
        return size < 0.05

    def filter_results(self, video_results, confidence_threshold, cuda=False):
        
        video_scores = video_results['scores']
        video_ind = (video_scores > confidence_threshold)
        video_ind = torch.logical_and(video_ind, self.get_relevant_ind(video_results['labels']))
        video_ind = torch.logical_and(video_ind, self.filter_large_bbox(video_results['boxes']))
        video_scores = video_scores[video_ind]
        video_bboxes = video_results['boxes'][video_ind, :]
        video_labels = video_results['labels'][video_ind]
        if cuda:
            return video_ind.cuda(), video_scores.cuda(), video_bboxes.cuda(), video_labels.cuda()
        else:
            return video_ind.cpu(), video_scores.cpu(), video_bboxes.cpu(), video_labels.cpu()
        
                

    def calc_accuracy(self, video, gt, args):
        '''
            Calculate the accuracy between video and gt using thresholds from args based on inference results
        '''

        assert video.keys() == gt.keys()

        f1s = []
        prs = []
        res = []

        for fid in video.keys():

            video_ind, video_scores, video_bboxes, video_labels = self.filter_results(video[fid], -1)
            gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(gt[fid], args.confidence_threshold)

            IoU = jaccard(video_bboxes, gt_bboxes)

            # let IoU = 0 if the label is wrong
            fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
            fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
            IoU[fat_video_labels != fat_gt_labels] = 0

            # calculate f1
            tp = 0
            
            for i in range(len(gt_labels)):
                 tp = tp + torch.min(self.step2(IoU[:, i] - args.iou_threshold), self.step2(video_scores - args.confidence_threshold)).max()
            fn = len(gt_labels) - tp
            fp = len(video_labels[video_scores > args.confidence_threshold]) - tp

            f1 = 2 * tp / (2 * tp + fp + fn)
            pr = tp / (tp + fp)
            re = tp / (tp + fn)

            f1s.append(f1)
            prs.append(pr)
            res.append(re)

            if fid % 10 == 9:
                print(torch.tensor(f1s[-9:]).mean())

        return {
            'f1': torch.tensor(f1s).mean().item(),
            'pr': torch.tensor(prs).mean().item(),
            're': torch.tensor(res).mean().item()
        }

    def calc_loss(self, video, gt_results, args):
        '''
            Inference and calculate the loss between video and gt using thresholds from args
        '''

        assert len(video.shape) == 4, f'The shape of video({video.shape}) must be 4D.'

        # calculate the ground truth
        gt_scores = gt_results['scores'].cuda()
        gt_ind = gt_scores > args.confidence_threshold
        gt_ind = torch.logical_and(gt_ind, self.get_relevant_ind(gt_results['labels'].cuda()))
        gt_bboxes = gt_results['boxes'][gt_ind, :].cuda()
        gt_labels = gt_results['labels'][gt_ind].cuda()

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
        return losses['loss_classifier'] + losses['loss_box_reg'], None

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

    def plot_results_on(self, gt_results, image, c, args):
        if gt_results == None:
            return image

        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(image)
        # load the cached results to cuda
        gt_scores = gt_results['scores']
        gt_ind = gt_scores > args.confidence_threshold
        gt_ind = torch.logical_and(gt_ind, self.get_relevant_ind(gt_results['labels']))
        gt_ind = torch.logical_and(gt_ind, self.filter_large_bbox(gt_results['boxes']))
        gt_bboxes = gt_results['boxes'][gt_ind, :]
        gt_labels = gt_results['labels'][gt_ind]

        for box in gt_bboxes:
            draw.rectangle(box.cpu().tolist(), width=4, outline=c)

        return image

        
    def get_undetected_ground_truth_index(self, gt, video, args):

        video_ind, video_scores, video_bboxes, video_labels = self.filter_results(video, args.confidence_threshold)
        gt_ind, gt_scores, gt_bboxes, gt_labels = self.filter_results(gt, args.confidence_threshold)

        # get IoU and clear the IoU of mislabeled objects
        IoU = jaccard(video_bboxes, gt_bboxes)
        fat_video_labels = video_labels[:, None].repeat(1, len(gt_labels))
        fat_gt_labels = gt_labels[None, :].repeat(len(video_labels), 1)
        IoU[fat_video_labels != fat_gt_labels] = 0

        return (IoU > args.iou_threshold).sum(dim=0) == 0

        