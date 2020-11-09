'''
    Compress the video through gradient-based optimization.
'''

from maskgen.fcn_16 import FCN
from utils.results_utils import read_results, read_ground_truth
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torchvision import io
import torchvision.transforms as T
import argparse
import coloredlogs
import logging
import enlighten
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import seaborn as sns
import glob
from pathlib import Path
import random
import os
from pdb import set_trace
sns.set()

offset = 6778


class TrainingDataset(Dataset):

    def __init__(self, paths):
        self.paths = paths
        self.nimages = len(glob.glob(f'{args.inputs[0]}/*.png'))

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):
        if offset <= idx < offset + 300:
            images = [plt.imread(f"{folder}/%05d.png" % idx)
                    for folder in args.inputs]
            images = [T.ToTensor()(image) for image in images]
        else:
            images = [1]
        return {
            'images': images,
            'fid': idx
        }


def get_loss(mask, target):

    # import pdb; pdb.set_trace()

    target = target.float()
    prob = mask.softmax(dim=1)[:, 1:2, :, :]
    prob = torch.where(target == 1, prob, 1-prob)
    weight = torch.where(
        target == 1, 10 * torch.ones_like(prob), torch.ones_like(prob))

    eps = 1e-6

    return (- weight * ((1-prob) ** 2) * ((prob+eps).log())).mean()


def main(args):

    # initialization for distributed training
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    # initialize logger
    logger = logging.getLogger('train')
    logger.addHandler(logging.FileHandler('train_youtube.log'))
    torch.set_default_tensor_type(torch.FloatTensor)

    # initalize bw
    bws = [0, 1]

    # construct dataset
    training_set = TrainingDataset(args.inputs)
    # training_set, _ = torch.utils.data.random_split(training_set, [int(
    #     0.5*len(training_set)), int(0.5*len(training_set))+1], generator=torch.Generator().manual_seed(100))
    training_set, cross_validation_set = torch.utils.data.random_split(training_set, [int(
        0.9*len(training_set)), int(0.1*len(training_set))+1], generator=torch.Generator().manual_seed(100))
    # training_sampler = torch.utils.data.DistributedSampler(training_set)
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    cross_validation_loader = torch.utils.data.DataLoader(
        cross_validation_set, batch_size=args.batch_size, num_workers=2
    )

    # construct the mask generator
    mask_generator = FCN()
    if os.path.exists(args.path):
        logger.info(f'Load the model from {args.path}')
        mask_generator.load(args.path)
    mask_generator.cuda()
    mask_generator.train()
    # mask_generator = torch.nn.parallel.DistributedDataParallel(mask_generator, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(
        mask_generator.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # load ground truth results
    ground_truth_results = read_ground_truth(args.ground_truth, logger)

    mean_cross_validation_loss_before = 100

    application = FasterRCNN_ResNet50_FPN()
    application.cuda()

    for iteration in range(args.num_iterations):

        '''
            Training
        '''

        progress_bar = enlighten.get_manager().counter(
            total=len(training_set), desc=f'Iteration {iteration} on training set', unit='frames')

        training_losses = []
        training_f1s = []

        for idx, data in enumerate(training_loader):

            progress_bar.update(incr=len(data['fid']))

            # extract data from dataloader
            images = data['images']
            if len(images[0].shape) != 4:
                continue
            images = [image.cuda() for image in images]
            fids = data['fid']
            fids = [fid.item() for fid in fids]

            with torch.no_grad():

                lq_image, hq_image = images[0], images[1]
                mask_generator_input = torch.cat(
                    [hq_image, hq_image - lq_image], dim=1)
                mask_slice = mask_generator(mask_generator_input)

                # calculate loss
                target = torch.cat(
                        [ground_truth_results[fid].long().cuda() for fid in fids])
                loss = get_loss(mask_slice, target)

                # optimization and logging
                logger.info(f'Training loss: %.3f', loss.item())
                training_losses.append(loss.item())

                mask_slice = mask_slice.softmax(dim=1)[:, 1:2, :, :]
                mask_slice[:, :, :, :] = torch.where(
                    mask_slice > percentile(mask_slice, 100-args.tile_percentage),
                    torch.ones_like(mask_slice),
                    torch.zeros_like(mask_slice))

                mask_slice = tile_mask(mask_slice, args.tile_size)
                mask_slice = binarize_mask(mask_slice, bws)
                masked_image = generate_masked_image(
                    mask_slice, images, bws)
                video_results = application.inference(
                    masked_image.cuda(), True)[0]
                ground_truth_result = application.inference(
                    hq_image.cuda(), True)[0]
                training_f1s.append(application.calc_accuracy({
                    fids[0]: video_results}, {
                        fids[0]: ground_truth_result}, args)['f1'])

        mean_training_loss=torch.tensor(training_losses).mean()
        mean_training_f1 = torch.tensor(training_f1s).mean()


        '''
            Cross validation
        '''

        progress_bar=enlighten.get_manager().counter(
            total = len(cross_validation_set), desc = f'Iteration {iteration} on cross validation set', unit = 'frames')

        cross_validation_losses=[]
        cross_validation_f1s = []

        for idx, data in enumerate(cross_validation_loader):

            progress_bar.update(incr = len(data['fid']))

            # extract data from dataloader
            images=data['images']
            if len(images[0].shape) != 4:
                continue
            images=[image.cuda() for image in images]
            fids=data['fid']
            fids=[fid.item() for fid in fids]

            # inference
            with torch.no_grad():
                lq_image, hq_image=images[0], images[1]
                mask_generator_input=torch.cat(
                    [hq_image, hq_image - lq_image], dim = 1)
                mask_slice=mask_generator(mask_generator_input)

                target=torch.cat(
                    [ground_truth_results[fid].long().cuda() for fid in fids])
                loss=get_loss(mask_slice, target)

                mask_slice = mask_slice.softmax(dim=1)[:, 1:2, :, :]
                mask_slice[:, :, :, :] = torch.where(
                    mask_slice > percentile(mask_slice, 100-args.tile_percentage),
                    torch.ones_like(mask_slice),
                    torch.zeros_like(mask_slice))

                mask_slice = tile_mask(mask_slice, args.tile_size)
                mask_slice = binarize_mask(mask_slice, bws)
                masked_image = generate_masked_image(
                    mask_slice, images, bws)
                video_results = application.inference(
                    masked_image.cuda(), True)[0]
                ground_truth_result = application.inference(
                    hq_image.cuda(), True)[0]
                cross_validation_f1s.append(application.calc_accuracy({
                    fids[0]: video_results}, {
                        fids[0]: ground_truth_result}, args)['f1'])

            # optimization and logging
            logger.info(f'Cross validation loss: {loss.item()}')
            cross_validation_losses.append(loss.item())

        mean_cross_validation_loss=torch.tensor(
            cross_validation_losses).mean().item()
        mean_cross_validation_f1 = torch.tensor(cross_validation_f1s).mean()
        logger.info('Average training loss: %.3f', mean_training_loss.item())
        logger.info('Average cross validation loss: %.3f',
                    mean_cross_validation_loss)
        logger.info('Average training f1: %.3f', mean_training_f1.item())
        logger.info('Average cross validation f1: %.3f',
                    mean_cross_validation_f1.item())

        break




if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(
        fmt = "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level = 'INFO')

    parser=argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs = '+',
                        help = 'The video file name. The largest video file will be the ground truth.', required = True)
    # parser.add_argument('-s', '--source', type=str, help='The original video source.', required=True)
    # parser.add_argument('-g', '--ground_truth', type=str,
    #                     help='The ground truth videos.', required=True)
    parser.add_argument('-p', '--path', type = str,
                        help = 'The path to store the generator parameters.', required = True)
    parser.add_argument('-g', '--ground_truth', type = str,
                        help = 'The ground truth file.', required = True)
    # parser.add_argument('-o', '--output', type=str,
    #                     help='The output name.', required=True)
    parser.add_argument('--confidence_threshold', type = float,
                        help = 'The confidence score threshold for calculating accuracy.', default = 0.5)
    parser.add_argument('--iou_threshold', type = float,
                        help = 'The IoU threshold for calculating accuracy in object detection.', default = 0.5)
    parser.add_argument('--num_iterations', type = int,
                        help = 'Number of iterations for optimizing the mask.', default = 500)
    parser.add_argument('--batch_size', type = int,
                        help = 'Number of iterations for optimizing the mask.', default = 2)
    parser.add_argument('--tile_size', type = int,
                        help = 'The tile size of the mask.', default = 8)
    parser.add_argument('--learning_rate', type = float,
                        help = 'The learning rate.', default = 1e-4)
    parser.add_argument('--gamma', type = float,
                        help = 'The gamma parameter for focal loss.', default = 2)
    # parser.add_argument('--mask_weight', type=float,
    #                     help='The weight of the mask normalization term', default=17)
    # parser.add_argument('--mask_p', type=int,
    #                     help='The p-norm for the mask.', default=1)
    # parser.add_argument('--binarize_weight', type=float,
    #                     help='The weight of the mask binarization loss term.', default=1)
    parser.add_argument('--local_rank', default = -1, type = int,
                        help = 'The GPU id for distributed training')
    parser.add_argument('--tile_percentage', type = float,
                        help = 'How many percentage of tiles will remain', default = 1)

    args=parser.parse_args()

    main(args)
