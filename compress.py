
import torch
from torchvision import io
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from PIL import Image
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pathlib import Path

from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
from utils.video_utils import read_videos, write_video, get_qp_from_name
from utils.mask_utils import *
from utils.results_utils import read_results


def main(args):

    # initialize
    logger = logging.getLogger('compress')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)
    videos = videos
    bws = bws
    qps = [get_qp_from_name(video_name) for video_name in video_names]
    
    
    # construct applications
    application_bundle = [FasterRCNN_ResNet50_FPN()]

    # construct the mask
    video_shape = videos[-1].shape
    mask_shape = [video_shape[0], 1, video_shape[2] // args.tile_size, video_shape[3] // args.tile_size]
    mask = torch.ones(mask_shape).float()
    mask.requires_grad = True

    optimizer = torch.optim.Adam([mask], lr=args.learning_rate)
    plt.clf()
    plt.figure(figsize=(16, 10))
    
    for iteration in range(args.num_iterations):

        optimizer.zero_grad()
        (args.mask_weight * mask.pow(args.mask_p).abs().mean()).backward()
        if args.cont_weight > 0:
            (args.cont_weight * (mask[1:, :, :, :] - mask[:-1, :, :, :]).abs().pow(args.cont_p).mean()).backward()

        binarized_mask = mask.clone().detach()
        binarize_mask(binarized_mask, bws)
        if iteration > args.num_iterations // 2:
            (args.binarize_weight * torch.sqrt(torch.tensor(iteration*1.0)) * (binarized_mask - mask).abs().pow(2).mean()).backward()
        

        for application in application_bundle:
            # read ground truth results
            ground_truth_results = read_results(args.ground_truth, application.name, logger)

            logger.info(f'Processing application {application.name}')
            progress_bar = enlighten.get_manager().counter(total=videos[-1].shape[0], desc=f'Iteration {iteration}: {application.name}', unit='frames')

            application.cuda()

            total_loss = []

            for fid, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(1) for video in videos]), mask.split(1))):

                progress_bar.update()

                # construct hybrid image
                mask_slice = tile_mask(mask_slice, args.tile_size)
                masked_image = generate_masked_image(mask_slice, video_slices, bws)

                # calculate the loss
                loss, video_results = application.calc_diff_acc(masked_image.cuda(), ground_truth_results[fid], args)
                loss.backward(retain_graph=True)
                total_loss.append(loss.item())

                # import pdb; pdb.set_trace()

                # visualization
                if fid % 5 == 0 and iteration % 10 == 9:
                    heat = tile_mask(mask[fid:fid+1, :, :, :], args.tile_size)[0, 0, :, :]
                    plt.clf()
                    ax = sns.heatmap(heat.detach().numpy(), zorder=3, alpha=0.5)
                    image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                    image = application.plot_results_on(ground_truth_results[fid], image, (255, 255, 255), args)
                    image = application.plot_results_on(video_results, image, (0, 255, 255), args)
                    ax.imshow(image, zorder=3, alpha=0.5)
                    Path(f'visualize/{args.output}/').mkdir(parents=True, exist_ok=True)
                    plt.savefig(f'visualize/{args.output}/{fid}_attn.png', bbox_inches='tight')
                

            application.cpu()
            
        optimizer.step()
        
        # clip mask
        mask_clip(mask, bws[0])

        logger.info('Loss: %0.3f, Mask max: %.3f, min: %.3f, mean: %.3f, std: %.3f' % (torch.tensor(total_loss).mean(), mask.max().item(), mask.min().item(), mask.mean().item(), mask.std().item()))

    # optimization done. No more gradients required.
    mask.requires_grad = False
    # write raw video
    masked_video = generate_masked_video(mask, videos, bws, args)
    write_masked_video(mask, args, qps, bws, logger)
    # write binarized video
    binarize_mask(mask, bws)
    write_video(masked_video, args.output, logger)


if __name__ == '__main__':

    # set the format of the logger
    coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", level='INFO')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs = '+', help='The video file names. The largest video file will be the ground truth.', required=True)
    parser.add_argument('-s', '--source', type=str, help='The original video source.', required=True)
    parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth results.', required=True)
    parser.add_argument('-o', '--output', type=str, help='The output name.', required=True)
    parser.add_argument('--confidence_threshold', type=float, help='The confidence score threshold for calculating accuracy.', default=0.3)
    parser.add_argument('--iou_threshold', type=float, help='The IoU threshold for calculating accuracy in object detection.', default=0.5)
    parser.add_argument('--num_iterations', type=int, help='Number of iterations for optimizing the mask.', default=30)
    parser.add_argument('--tile_size', type=int, help='The tile size of the mask.', default=16)
    parser.add_argument('--learning_rate', type=float, help='The learning rate.', default=0.04)
    parser.add_argument('--mask_weight', type=float, help='The weight of the mask normalization term', default=128)
    parser.add_argument('--mask_p', type=int, help='The p-norm for the mask.', default=1)
    parser.add_argument('--binarize_weight', type=float, help='The weight of the mask binarization loss term.', default=1)
    parser.add_argument('--cont_weight', type=float, help='The weight of the continuity normalization term', default=0)
    parser.add_argument('--cont_p', type=int, help='The p-norm for the continuity.', default=1)


    args = parser.parse_args()

    main(args)
