

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import logging
from .dnn import DNN

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

    def calc_accuracy(self, video, gt):

        assert video.shape == gt.shape, f'The shape of video({video.shape}) and gt({gt.shape}) must be the same in order to calculate the accuracy'

        video_results, gt_results = self.inference(torch.cat([video, gt]))

        