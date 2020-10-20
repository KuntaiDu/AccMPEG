'''
    Compress the video through gradient-based optimization.
'''

from maskgen.fcn import FCN
from utils.results_utils import read_results
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import io
import torchvision.transforms as T
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import seaborn as sns
import glob
from pathlib import Path
import random
import os
sns.set()

class TrainingDataset(Dataset):

    def __init__(self, paths):
        self.paths = paths
        self.nimages = len(glob.glob(f'{args.inputs[0]}/*.png'))

    def __len__(self):
        return self.nimages
    
    def __getitem__(self, idx):
        images = [plt.imread(f"{folder}/%05d.png" % idx) for folder in args.inputs]
        images = [T.ToTensor()(image) for image in images]
        return {
            'images': images,
            'fid': idx
        }


def main(args):

    # initialize
    logger = logging.getLogger('train')
    logger.addHandler(logging.FileHandler('train.log'))
    torch.set_default_tensor_type(torch.FloatTensor)

    # initalize bw
    bws = [0, 1]

    # construct dataset
    training_set = TrainingDataset(args.inputs)


    # construct applications
    application_bundle = [FasterRCNN_ResNet50_FPN()]

    # construct the mask generator
    mask_generator = FCN()
    if os.path.exists(args.path):
        logger.info(f'Load the model from {args.path}')
        mask_generator.load(args.path)
    mask_generator.cuda()
    mask_generator.train()
    
    optimizer = torch.optim.Adam(
        mask_generator.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    ground_truth_results = {}
    for iteration in range(args.num_iterations):
        for application in application_bundle:

            logger.info(f'Processing application {application.name}')
            progress_bar = enlighten.get_manager().counter(
                total=len(training_set), desc=f'Iteration {iteration}: {application.name}', unit='frames')

            application.cuda()


            application_losses = []
            sparsity_losses = []
            total_losses = []

            training_loader = DataLoader(training_set, batch_size = 1, shuffle=True, num_workers = 1)

            for idx, data in enumerate(training_loader):
                
                images = data['images']
                images = [image.cuda() for image in images]
                fid = data['fid'].item()

                progress_bar.update()

                if fid not in ground_truth_results.keys():
                    ground_truth_results[fid] = application.inference(images[-1], detach=True)[0]
                # import pdb; pdb.set_trace()

                # construct hybrid image
                mask_slice = mask_generator(images[-1])
                mask_slice_tile = tile_mask(mask_slice, args.tile_size)
                masked_image = generate_masked_image(
                    mask_slice_tile, images, bws)

                # calculate the loss
                application_loss, video_results = application.calc_loss(
                    masked_image.cuda(), ground_truth_results[fid], args)

                sparsity_loss = (1 / 60) * args.mask_weight * mask_slice.pow(args.mask_p).abs().mean()
                total_loss = application_loss + sparsity_loss

                total_loss.backward()

                application_losses.append(application_loss.item())
                sparsity_losses.append(sparsity_loss.item())
                total_losses.append(total_loss.item())

                if len(application_losses) == args.batch_size:
                    logger.info('APP loss:%0.3f, SPA loss:%0.3f, L1 norm:%0.3f, TOT loss:%0.3f', torch.tensor(application_losses).mean(), torch.tensor(sparsity_losses).mean(), mask_slice.mean(), torch.tensor(application_losses).mean()+torch.tensor(sparsity_losses).mean())
                    application_losses = []
                    sparsity_losses = []
                    optimizer.step()
                    optimizer.zero_grad()

                # import pdb; pdb.set_trace()

                # # visualization
                # if fid % 5 == 0 and (iteration+1) % (args.num_iterations // 4) == 0:
                #     heat = tile_mask(
                #         mask[fid:fid+1, :, :, :], args.tile_size)[0, 0, :, :]
                #     plt.clf()
                #     ax = sns.heatmap(heat.detach().numpy(),
                #                      zorder=3, alpha=0.5)
                #     image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                #     image = application.plot_results_on(
                #         ground_truth_results[fid], image, (255, 255, 255), args)
                #     image = application.plot_results_on(
                #         video_results, image, (0, 255, 255), args)
                #     ax.imshow(image, zorder=3, alpha=0.5)
                #     Path(
                #         f'visualize/{args.output}/').mkdir(parents=True, exist_ok=True)
                #     plt.savefig(
                #         f'visualize/{args.output}/{fid}_attn.png', bbox_inches='tight')

            scheduler.step(torch.tensor(total_losses).mean())

            application.cpu()

        mask_generator.save(args.path)


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs='+',
                        help='The video file name. The largest video file will be the ground truth.', required=True)
    # parser.add_argument('-s', '--source', type=str, help='The original video source.', required=True)
    # parser.add_argument('-g', '--ground_truth', type=str,
    #                     help='The ground truth videos.', required=True)
    parser.add_argument('-p', '--path', type=str,
                        help = 'The path to store the generator parameters.', required=True)
    # parser.add_argument('-o', '--output', type=str,
    #                     help='The output name.', required=True)
    parser.add_argument('--confidence_threshold', type=float,
                        help='The confidence score threshold for calculating accuracy.', default=0.5)
    parser.add_argument('--iou_threshold', type=float,
                        help='The IoU threshold for calculating accuracy in object detection.', default=0.5)
    parser.add_argument('--num_iterations', type=int,
                        help='Number of iterations for optimizing the mask.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of iterations for optimizing the mask.', default=4)
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=8)
    parser.add_argument('--learning_rate', type=float,
                        help='The learning rate.', default=1e-3)
    parser.add_argument('--mask_weight', type=float,
                        help='The weight of the mask normalization term', default=17)
    parser.add_argument('--mask_p', type=int,
                        help='The p-norm for the mask.', default=1)
    parser.add_argument('--binarize_weight', type=float,
                        help='The weight of the mask binarization loss term.', default=1)

    args = parser.parse_args()

    main(args)
