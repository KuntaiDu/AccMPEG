'''
    Compress the video through gradient-based optimization.
'''

from utils.results_utils import read_results
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import io
from maskgen.fcn import FCN
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from PIL import Image
import seaborn as sns
sns.set()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger('test')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = [0, 1]
    qps = [get_qp_from_name(video_name) for video_name in video_names]

    # construct applications
    application_bundle = [FasterRCNN_ResNet50_FPN()]

    mask_generator = FCN()
    mask_generator.load(args.path)
    mask_generator.eval().cuda()

    # construct the mask
    video_shape = videos[-1].shape
    mask_shape = [video_shape[0], 1, video_shape[2] //
                  args.tile_size, video_shape[3] // args.tile_size]
    mask = torch.ones(mask_shape).float()

    ground_truth_dict = read_results(args.inputs[-1], 'FasterRCNN_ResNet50_FPN', logger)

    # binarized_mask = mask.clone().detach()
    # binarize_mask(binarized_mask, bws)
    # if iteration > 3 * (args.num_iterations // 4):
    #     (args.binarize_weight * torch.tensor(iteration*1.0) * (binarized_mask - mask).abs().pow(2).mean()).backward()

    for application in application_bundle:

        logger.info(f'Processing application {application.name}')
        progress_bar = enlighten.get_manager().counter(
            total=videos[-1].shape[0], desc=f'{application.name}', unit='frames')

        application.cuda()

        f1_list = []

        for fid, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(1) for video in videos]), mask.split(1))):

            progress_bar.update()

            # construct hybrid image
            with torch.no_grad():
                mask_slice[:, :, :, :] = mask_generator(
                    video_slices[-1].cuda())

            # calculate the loss, to see the generalization error
            with torch.no_grad():
                mask_slice = binarize_mask(tile_mask(mask_slice, args.tile_size), bws)
                masked_image = generate_masked_image(
                    mask_slice, video_slices, bws)

                video_results = application.inference(
                    masked_image.cuda(), True)[0]
                f1_list.append(application.calc_accuracy({
                    fid: video_results
                }, {
                    fid: ground_truth_dict[fid]
                }, args)['f1'])

                # import pdb; pdb.set_trace()
                # loss, _ = application.calc_loss(masked_image.cuda(),
                #                                 application.inference(video_slices[-1].cuda(), detach=True)[0], args)
                # total_loss.append(loss.item())

            # visualization
            if fid % 30 == 29:
                logger.info('The average f1 on 30 frames is %.3f' %
                            torch.tensor(f1_list[-29:]).mean())
                heat = tile_mask(mask[fid:fid+1, :, :, :],
                                 args.tile_size)[0, 0, :, :]
                plt.clf()
                ax = sns.heatmap(heat.detach().numpy(), zorder=3, alpha=0.5)
                image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                #image = application.plot_results_on(ground_truth_results[fid], image, (255, 255, 255), args)
                #image = application.plot_results_on(video_results, image, (0, 255, 255), args)
                ax.imshow(image, zorder=3, alpha=0.5)
                Path(
                    f'visualize/{args.output}/').mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    f'visualize/{args.output}/{fid}_attn.png', bbox_inches='tight')

        logger.info('The average f1 is %.3f' % torch.tensor(f1_list).mean())

        application.cpu()

    mask.requires_grad = False
    mask = binarize_mask(mask, bws)
    write_masked_video(mask, args, qps, bws, logger)
    # masked_video = generate_masked_video(mask, videos, bws, args)
    # write_video(masked_video, args.output, logger)


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs='+',
                        help='The video file names. The largest video file will be the ground truth.', required=True)
    parser.add_argument('-s', '--source', type=str,
                        help='The original video source.', required=True)
    # parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth results.', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='The output name.', required=True)
    parser.add_argument('--confidence_threshold', type=float,
                        help='The confidence score threshold for calculating accuracy.', default=0.5)
    parser.add_argument('--iou_threshold', type=float,
                        help='The IoU threshold for calculating accuracy in object detection.', default=0.5)
    parser.add_argument('--num_iterations', type=int,
                        help='Number of iterations for optimizing the mask.', default=30)
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=16)
    parser.add_argument('--learning_rate', type=float,
                        help='The learning rate.', default=0.04)
    parser.add_argument('--mask_weight', type=float,
                        help='The weight of the mask normalization term', default=128)
    parser.add_argument('--mask_p', type=int,
                        help='The p-norm for the mask.', default=1)
    parser.add_argument('--binarize_weight', type=float,
                        help='The weight of the mask binarization loss term.', default=1)
    parser.add_argument('--cont_weight', type=float,
                        help='The weight of the continuity normalization term', default=0)
    parser.add_argument('--cont_p', type=int,
                        help='The p-norm for the continuity.', default=1)
    parser.add_argument('-p', '--path', type=str,
                        help='The path to store the generator parameters.', required=True)

    args = parser.parse_args()

    main(args)
