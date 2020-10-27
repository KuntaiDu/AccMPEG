'''
    Generate ground truth for image files.
'''

from utils.results_utils import read_results
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import io
from torch.utils.data import Dataset, DataLoader
from torch.distributions.bernoulli import Bernoulli
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from PIL import Image
import seaborn as sns
sns.set()


class TrainingDataset(Dataset):

    def __init__(self, paths):
        self.paths = paths
        self.nimages = len(glob.glob(f'{args.inputs[0]}/*.png'))

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):
        images = [plt.imread(f"{folder}/%05d.png" % idx)
                  for folder in args.inputs]
        images = [T.ToTensor()(image) for image in images]
        return {
            'images': images,
            'fid': idx
        }


def main(args):

    # initialize
    logger = logging.getLogger('compress')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    torch.set_default_tensor_type(torch.FloatTensor)

    # # read the video frames (will use the largest video as ground truth)
    # videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    # videos = videos
    # bws = [0, 1]
    # qps = [get_qp_from_name(video_name) for video_name in video_names]

    # initialize bw
    bws = [0, 1]

    # construct dataset
    training_set = TrainingDataset(args.inputs)
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # construct applications
    application = FasterRCNN_ResNet50_FPN()
    application.cuda()

    # # construct the mask
    # video_shape = videos[-1].shape
    # num_frames = video_shape[0]
    # mask_shape = [num_frames, 1, video_shape[2] //
    #               args.tile_size, video_shape[3] // args.tile_size]
    # assert num_frames % args.batch_size == 0
    # sum_grad = torch.zeros(mask_shape)
    # neg_grad_exp = (-sum_grad * args.learning_rate).exp()
    # mask = torch.zeros(mask_shape)
    # mask.requires_grad = True

    # plt.clf()
    # plt.figure(figsize=(16, 10))

    progress_bar = enlighten.get_manager().counter(
        total=len(training_set), desc=f'{application.name}', unit='frames')
    mask_results = []

    for batch_id, data in enumerate(training_loader):
        progress_bar.update(args.batch_size)
        logger.info('Processing batch %d', batch_id)

        images = data['images']
        images = [image.cuda() for image in images]
        fids = data['fid']
        fids = [fid.item() for fid in fids]

        # get ground truth results
        ground_truth_results = [application.inference(
            images_slice, detach=True)[0] for images_slice in images[-1].split(1)]

        # construct mask
        mask_shape = [images[-1].shape[0], images[-1].shape[1], images[-1].shape[2] //
                      args.tile_size, images[-1].shape[3] // args.tile_size]
        mask = torch.zeros(mask_shape).cuda()
        sum_grad = torch.zeros(mask.shape).cuda()
        mask.requires_grad = True

        for i in range(args.num_iterations):

            # generate masked image
            masked_image = generate_masked_video(mask, images, bws, args)

            # calculate application loss
            application_loss = application.calc_loss(
                masked_image.cuda(),
                ground_truth_results,
                args
            )
            application_loss.backward(retain_graph=True)

            # calculate new mask
            mask.requires_grad = False
            sum_grad += mask.grad
            neg_grad = -sum_grad
            mask = torch.where(neg_grad > percentile(neg_grad, 100-args.tile_percentage),
                            torch.ones_like(neg_grad),
                            torch.zeros_like(neg_grad))
            mask.requires_grad = True

        with open(args.output, 'a+b') as f:
            for fid, mask_slice in zip(fids, mask.split(1)):
                pickle.dump({fid: mask_slice}, f)

if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs='+',
                        help='The video file names. The largest video file will be the ground truth.', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='The output name.', required=True)
    parser.add_argument('--confidence_threshold', type=float,
                        help='The confidence score threshold for calculating accuracy.', default=0.5)
    parser.add_argument('--iou_threshold', type=float,
                        help='The IoU threshold for calculating accuracy in object detection.', default=0.5)
    parser.add_argument('--num_iterations', type=int,
                        help='Number of iterations for optimizing the mask.', default=6)
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=8)
    parser.add_argument('--batch_size', type=int,
                        help='The batch size', default=2)
    parser.add_argument('--tile_percentage', type=float,
                        help='How many percentage of tiles will remain', default=1)
    # parser.add_argument('--mask_p', type=int, help='The p-norm for the mask.', default=1)
    # parser.add_argument('--binarize_weight', type=float, help='The weight of the mask binarization loss term.', default=1)
    # parser.add_argument('--cont_weight', type=float, help='The weight of the continuity normalization term', default=0)
    # parser.add_argument('--cont_p', type=int, help='The p-norm for the continuity.', default=1)

    args = parser.parse_args()

    main(args)
