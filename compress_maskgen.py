'''
    Compress the video through gradient-based optimization.
'''

from utils.results_utils import read_results, read_ground_truth
from utils.mask_utils import *
from utils.video_utils import read_videos, write_video, get_qp_from_name
from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import io
from maskgen.fcn_16 import FCN
import argparse
import coloredlogs
import logging
import enlighten
import torchvision.transforms as T
from PIL import Image
import seaborn as sns
sns.set()


def get_loss(mask, target):

    # import pdb; pdb.set_trace()

    target = target.float().cuda()
    prob = mask.softmax(dim=1)[:, 1:2, :, :]
    prob = torch.where(target == 1, prob, 1-prob)
    weight = torch.where(
        target == 1, 10 * torch.ones_like(prob), torch.ones_like(prob))

    eps = 1e-6

    return (- weight * ((1-prob) ** 2) * ((prob+eps).log())).mean()


def main(args):

    gc.enable()

    # initialize
    logger = logging.getLogger('maskgen')
    logger.addHandler(logging.FileHandler('maskgen.log'))
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

    ground_truth_dict = read_results(
        args.inputs[-1], 'FasterRCNN_ResNet50_FPN', logger)

    logger.info('Reading ground truth mask')
    with open(args.mask + '.mask', 'rb') as f:
        ground_truth_mask = pickle.load(f)
    ground_truth_mask = ground_truth_mask[sorted(ground_truth_mask.keys())[1]]
    ground_truth_mask = ground_truth_mask.split(1)

    plt.clf()
    plt.figure(figsize=(16, 10))

    # binarized_mask = mask.clone().detach()
    # binarize_mask(binarized_mask, bws)
    # if iteration > 3 * (args.num_iterations // 4):
    #     (args.binarize_weight * torch.tensor(iteration*1.0) * (binarized_mask - mask).abs().pow(2).mean()).backward()

    for application in application_bundle:

        logger.info(f'Processing application {application.name}')
        progress_bar = enlighten.get_manager().counter(
            total=videos[-1].shape[0], desc=f'{application.name}', unit='frames')

        application.cuda()

        losses = []
        f1s = []

        for fid, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(1) for video in videos]), mask.split(1))):

            progress_bar.update()

            lq_image, hq_image = video_slices[0], video_slices[1]
            # lq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_34/%05d.png' % (fid+offset2)))[None, :, :, :]

            # construct hybrid image
            with torch.no_grad():
                mask_gen = mask_generator(
                    torch.cat([hq_image, hq_image - lq_image], dim=1).cuda())
                losses.append(get_loss(mask_gen, ground_truth_mask[fid]))
                mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
                mask_slice[:, :, :, :] = torch.where(
                    mask_gen > percentile(mask_gen, 100-args.tile_percentage),
                    torch.ones_like(mask_gen),
                    torch.zeros_like(mask_gen))
                # mask_slice[:, :, :, :] = torch.where(mask_gen > 0.5, torch.ones_like(mask_gen), torch.zeros_like(mask_gen))
            # mask_slice[:, :, :, :] = ground_truth_mask[fid + offset2].float()

            # calculate the loss, to see the generalization error
            with torch.no_grad():
                mask_slice = tile_mask(mask_slice, args.tile_size)
                masked_image = generate_masked_image(
                    mask_slice, video_slices, bws)

                video_results = application.inference(
                    masked_image.cuda(), True)[0]
                f1s.append(application.calc_accuracy({
                    fid: video_results
                }, {
                    fid: ground_truth_dict[fid]
                }, args)['f1'])

            # import pdb; pdb.set_trace()
            # loss, _ = application.calc_loss(masked_image.cuda(),
            #                                 application.inference(video_slices[-1].cuda(), detach=True)[0], args)
            # total_loss.append(loss.item())

            # visualization
            # if fid % 30 == 0:
            #     heat = tile_mask(mask_slice,
            #                      args.tile_size)[0, 0, :, :]
            #     plt.clf()
            #     ax = sns.heatmap(heat.detach().numpy(), zorder=3, alpha=0.5)
            #     # hq_image = T.ToTensor()(Image.open('youtube_videos/train_pngs_qp_24/%05d.png' % (fid+offset2)))[None, :, :, :].cuda()
            #     # with torch.no_grad():
            #     #     inf = application.inference(hq_image, detach=True)[0]
            #     image = T.ToPILImage()(video_slices[-1][0, :, :, :])
            #     # image = application.plot_results_on(inf, image, (255, 255, 255), args)
            #     #image = application.plot_results_on(video_results, image, (0, 255, 255), args)
            #     ax.imshow(image, zorder=3, alpha=0.5)
            #     Path(
            #         f'visualize/{args.output}/').mkdir(parents=True, exist_ok=True)
            #     plt.savefig(
            #         f'visualize/{args.output}/{fid}_attn.png', bbox_inches='tight')

        logger.info('In video %s', args.output)
        logger.info('The average loss is %.3f' % torch.tensor(losses).mean())
        logger.info('The average f1 is %.3f' % torch.tensor(f1s).mean())

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
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=8)
    parser.add_argument('-p', '--path', type=str,
                        help='The path of pth file that stores the generator parameters.', required=True)
    parser.add_argument('--tile_percentage', type=float,
                        help='How many percentage of tiles will remain', default=1)
    parser.add_argument('--mask', type=str,
                        help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
