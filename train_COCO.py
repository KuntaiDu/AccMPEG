"""
    Train the NN-basedmask generator.
"""

import argparse
import glob
import importlib.util
import logging
import math
import os
import random
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import io
from torchvision.datasets import CocoDetection

from dnn.dnn_factory import DNN_Factory

# from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
# from dnn.fcn_resnet50 import FCN_ResNet50
from utils.bbox_utils import center_size
from utils.loss_utils import cross_entropy_thresh as get_loss
from utils.mask_utils import *
from utils.results_utils import read_results
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import visualize_heat

sns.set()

thresh_list = [0.1, 0.01]
weight = [1, 1]

path2data = "/tank/kuntai/COCO_Detection/train2017"
path2json = "/tank/kuntai/COCO_Detection/annotations/instances_train2017.json"


# def transform(image):
#     w, h = image.size
#     padh = (h + args.tile_size - 1) // args.tile_size * args.tile_size - h
#     padw = (w + args.tile_size - 1) // args.tile_size * args.tile_size - w
#     pad = T.Pad((0, 0, padh, padw), fill=(123, 116, 103))
#     return T.ToTensor()(pad(image))


class COCO_Dataset(Dataset):
    def __init__(self):
        self.path = "/tank/kuntai/COCO_Detection/train2017_reorder/"
        self.len = len(glob.glob(self.path + "*.jpg"))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.path + "%010d.jpg" % idx).convert("RGB")

        w, h = image.size
        if h > w:
            return None
        transform_hq = T.Compose(
            [
                # T.Pad(
                #     (
                #         math.floor((1280 - w) / 2),
                #         math.floor((720 - h) / 2),
                #         math.ceil((1280 - w) / 2),
                #         math.ceil((720 - h) / 2),
                #     ),
                #     fill=(123, 116, 103),
                # ),
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )
        transform_lq = T.Compose(
            [
                # T.Pad(
                #     (
                #         math.floor((1280 - w) / 2),
                #         math.floor((720 - h) / 2),
                #         math.ceil((1280 - w) / 2),
                #         math.ceil((720 - h) / 2),
                #     ),
                #     fill=(123, 116, 103),
                # ),
                T.Resize((272, 480)),
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )

        return {
            "hq": transform_hq(image),
            "lq": transform_lq(image),
            "fid": idx,
        }


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) >= 1:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None


def main(args):

    # initialization for distributed training
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    # initialize logger
    logger = logging.getLogger("train_COCO")
    logger.addHandler(logging.FileHandler(args.log))
    torch.set_default_tensor_type(torch.FloatTensor)
    train_writer = SummaryWriter("runs/train")
    cross_writer = SummaryWriter("runs/cross")

    # construct training set and cross validation set
    train_val_set = COCO_Dataset()
    # train_val_set, _ = torch.utils.data.random_split(
    #     train_val_set,
    #     [math.ceil(0.1 * len(train_val_set)), math.floor(0.9 * len(train_val_set))],
    #     generator=torch.Generator().manual_seed(100),
    # )
    training_set, cross_validation_set = torch.utils.data.random_split(
        train_val_set,
        [
            math.ceil(0.7 * len(train_val_set)),
            math.floor(0.3 * len(train_val_set)),
        ],
        generator=torch.Generator().manual_seed(100),
    )
    # training_sampler = torch.utils.data.DistributedSampler(training_set)
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        collate_fn=my_collate,
    )
    cross_validation_loader = torch.utils.data.DataLoader(
        cross_validation_set,
        batch_size=args.batch_size,
        num_workers=10,
        collate_fn=my_collate,
    )

    # construct the mask generator
    maskgen_spec = importlib.util.spec_from_file_location(
        "maskgen", args.maskgen_file
    )
    maskgen = importlib.util.module_from_spec(maskgen_spec)
    maskgen_spec.loader.exec_module(maskgen)
    mask_generator = maskgen.FCN()
    if os.path.exists(args.init):
        logger.info(f"Load the model from %s", args.init)
        mask_generator.load(args.init)
    mask_generator.train()
    # mask_generator = nn.DataParallel(mask_generator)

    # mask_generator = torch.nn.parallel.DistributedDataParallel(mask_generator, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(
        mask_generator.parameters(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # load ground truth results
    saliency = {}

    if False and len(glob.glob(args.ground_truth + "*")) != 0:
        saliency = {}
        for ground_truth in glob.glob(args.ground_truth + "*"):
            with open(ground_truth, "rb") as f:
                saliency.update(pickle.load(f))
    else:
        # get the application
        # generate saliency
        app = DNN_Factory().get_model(args.app)
        loader = torch.utils.data.DataLoader(
            train_val_set, shuffle=False, num_workers=4, collate_fn=my_collate
        )
        progress_bar = enlighten.get_manager().counter(
            total=len(train_val_set),
            desc=f"Generating saliency as ground truths",
            unit="frames",
        )
        saliency = {}

        saliency = {}
        for ground_truth in glob.glob(args.ground_truth + "*"):
            with open(ground_truth, "rb") as f:
                saliency.update(pickle.load(f))

        for data in loader:
            progress_bar.update()
            # get data
            if data == None:
                continue
            fid = data["fid"].item()
            # if fid % 3 != args.local_rank:
            #     continue
            hq_image = data["hq"]
            # lq_image = data["lq"].cuda(non_blocking=True)
            lq_image = (
                torch.ones_like(hq_image)
                * torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
            )
            hq_image = hq_image.cuda(non_blocking=True)
            lq_image = lq_image.cuda(non_blocking=True)
            lq_image.requires_grad = True

            with torch.no_grad():
                hq_result = app.inference(hq_image, detach=True)
                hq_result = app.filter_result(hq_result, args)
            if len(hq_result["instances"]) == 0 and fid in saliency:
                del saliency[fid]

            # with torch.enable_grad():
            #     loss = app.calc_loss(lq_image, hq_result, args)
            # # lq_image.requires_grad = True
            # # # print(lq_image.requires_grad)
            # # with torch.enable_grad():
            # #     lq_result = application.model(lq_image)["out"]
            # #     loss = F.cross_entropy(lq_result, hq_result)
            # #     # print(lq_image.requires_grad)
            # loss.backward()

            # mask_grad = lq_image.grad.norm(dim=1, p=2, keepdim=True)
            # mask_grad = F.conv2d(
            #     mask_grad,
            #     torch.ones([1, 1, args.tile_size, args.tile_size]).cuda(),
            #     stride=args.tile_size,
            # )
            # # determine the threshold
            # mask_grad = mask_grad.detach().cpu()
            # # normalize gradient to [0, 1]
            # mask_grad = mask_grad - mask_grad.min()
            # mask_grad = mask_grad / mask_grad.max()
            # mask_grad = mask_grad.detach().cpu()

            # # save it
            # saliency[fid] = mask_grad.detach().cpu()

            # visualize the saliency
            if False and fid % 500 == 0:

                # visualize
                if args.visualize:
                    image = T.ToPILImage()(data["hq"][0])
                    # application.plot_results_on(
                    #     hq_result[0].cpu(), image, "Azure", args, train=True
                    # )
                    image = app.visualize(image, hq_result, args)

                    # plot the ground truth
                    visualize_heat(
                        image,
                        mask_grad,
                        f"train/{args.path}/{fid}_saliency.png",
                        args,
                    )

                    # # visualize distribution
                    # fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)

                    # try:
                    #     sns.distplot(sum_mask.flatten().detach().numpy())
                    #     fig.savefig(
                    #         f"train/{args.path}/{fid}_logdist.png", bbox_inches="tight"
                    #     )
                    # except:
                    #     pass
                    # plt.close(fig)

                    # # write mean and std in gaussian mixture model
                    # with open(f"train/{args.path}/{fid}_mean_std.txt", "w") as f:
                    #     f.write(f"{mean} {std}")

        # write saliency to disk
        with open(args.ground_truth + f"{args.local_rank}", "wb") as f:
            pickle.dump(saliency, f)

    # training
    mask_generator.cuda()
    mean_cross_validation_loss_before = 100

    for iteration in range(args.num_iterations):

        """
            Training
        """

        progress_bar = enlighten.get_manager().counter(
            total=len(training_set),
            desc=f"Iteration {iteration} on training set",
            unit="frames",
        )

        training_losses = []

        for idx, data in enumerate(training_loader):

            progress_bar.update(incr=args.batch_size)

            # inference
            # if not any("bbox" in _ for _ in data[1]):
            #     continue
            # fids = [data[1][0]["image_id"].item()]
            # if fids[0] not in saliency[thresholds[0]]:
            #     continue
            if data == None:
                continue
            fids = [fid.item() for fid in data["fid"]]
            if any(fid not in saliency for fid in fids):
                continue
            target = torch.cat([saliency[fid] for fid in fids]).cuda(
                non_blocking=True
            )
            hq_image = data["hq"].cuda(non_blocking=True)
            mask_slice = mask_generator(hq_image)

            # calculate loss
            loss = get_loss(mask_slice, target, thresh_list)
            loss.backward()

            # optimization and logging
            if idx % 1 == 0:
                train_writer.add_scalar(
                    Path(args.path).stem,
                    loss.item(),
                    idx
                    + iteration
                    * (len(training_set) + len(cross_validation_set)),
                )
            training_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            if any(fid % 500 == 0 for fid in fids):
                # save the model
                mask_generator.save(args.path)
                # visualize
                if args.visualize:
                    maxid = np.argmax([fid % 500 == 0 for fid in fids]).item()
                    fid = fids[maxid]
                    image = T.ToPILImage()(data["hq"][maxid])
                    mask_slice = mask_slice[maxid : maxid + 1, :, :, :]
                    mask_slice = mask_slice.softmax(dim=1)[:, 1:2, :, :]
                    target = target[maxid : maxid + 1, :, :, :]
                    target = sum(
                        (target > thresh).float() for thresh in thresh_list
                    )
                    # hq_image.requires_grad = True
                    # get salinecy
                    # gt_result = application.inference(hq_image.cuda(), nograd=False)[0]
                    # _, scores, boxes, _ = application.filter_results(
                    #     gt_result, args.confidence_threshold, True, train=True
                    # )
                    # sums = scores.sum()
                    # sums.backward()
                    visualize_heat(
                        image,
                        mask_slice.cpu().detach(),
                        f"train/{args.path}/{fid}_train.png",
                        args,
                    )

                    visualize_heat(
                        image,
                        target.cpu().detach(),
                        f"train/{args.path}/{fid}_saliency.png",
                        args,
                    )
                    # application.plot_results_on(
                    #     gt_result, image, "Azure", args, train=True
                    # )
                    # fid = fids[0]

                    # plot the ground truth
                    # if not Path(f"train/{args.path}/{fid}_train.png").exists():
                    #     fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
                    #     sum_mask = tile_mask(
                    #         sum(saliency[thresh][fid].float() for thresh in thresholds),
                    #         args.tile_size,
                    #     )[0, 0, :, :]
                    #     ax = sns.heatmap(
                    #         sum_mask.cpu().detach().numpy(),
                    #         zorder=3,
                    #         alpha=0.5,
                    #         ax=ax,
                    #         xticklabels=False,
                    #         yticklabels=False,
                    #     )
                    #     ax.imshow(image, zorder=3, alpha=0.5)
                    #     ax.tick_params(left=False, bottom=False)
                    #     Path(f"train/{args.path}/").mkdir(parents=True, exist_ok=True)
                    #     fig.savefig(
                    #         f"train/{args.path}/{fid}_train.png", bbox_inches="tight"
                    #     )
                    #     plt.close(fig)

                    # visualize the test mask
                    # fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
                    # sum_mask = tile_mask(mask_slice_temp, args.tile_size,)[0, 0, :, :]
                    # ax = sns.heatmap(
                    #     sum_mask.cpu().detach().numpy(),
                    #     zorder=3,
                    #     alpha=0.5,
                    #     ax=ax,
                    #     xticklabels=False,
                    #     yticklabels=False,
                    # )
                    # ax.imshow(image, zorder=3, alpha=0.5)
                    # ax.tick_params(left=False, bottom=False)
                    # Path(f"train/{args.path}/").mkdir(parents=True, exist_ok=True)
                    # fig.savefig(
                    #     f"train/{args.path}/{fid}_test.png", bbox_inches="tight"
                    # )
                    # plt.close(fig)

                    # mask_grad = hq_image.grad.norm(dim=1, p=2, keepdim=True)
                    # mask_grad = F.conv2d(
                    #     mask_grad,
                    #     torch.ones([1, 1, args.tile_size, args.tile_size]).cuda(),
                    #     stride=args.tile_size,
                    # )
                    # mask_grad = tile_mask(mask_grad, args.tile_size)
                    # fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
                    # sum_mask = mask_grad[0, 0, :, :].log().cpu().detach()
                    # ax = sns.heatmap(
                    #     sum_mask.numpy(),
                    #     zorder=3,
                    #     alpha=0.5,
                    #     ax=ax,
                    #     xticklabels=False,
                    #     yticklabels=False,
                    # )
                    # ax.imshow(image, zorder=3, alpha=0.5)
                    # ax.tick_params(left=False, bottom=False)
                    # Path(f"train/{args.path}/").mkdir(parents=True, exist_ok=True)
                    # fig.savefig(
                    #     f"train/{args.path}/{fid}_saliency.png", bbox_inches="tight"
                    # )
                    # plt.close(fig)

                    # fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
                    # sns.distplot(sum_mask.flatten().detach().numpy())
                    # fig.savefig(
                    #     f"train/{args.path}/{fid}_logdist.png", bbox_inches="tight"
                    # )
                    # plt.close(fig)

        mean_training_loss = torch.tensor(training_losses).mean()
        logger.info("Average training loss: %.3f", mean_training_loss.item())

        """
            Cross validation
        """

        progress_bar = enlighten.get_manager().counter(
            total=len(cross_validation_set),
            desc=f"Iteration {iteration} on cross validation set",
            unit="frames",
        )

        cross_validation_losses = []

        for idx, data in enumerate(cross_validation_loader):

            progress_bar.update(incr=args.batch_size)

            # # extract data from dataloader
            # if not any("bbox" in _ for _ in data[1]):
            #     continue
            # fids = [data[1][0]["image_id"].item()]

            # if fids[0] not in saliency[thresholds[0]]:
            #     continue
            # hq_image = data[0].cuda()

            if data == None:
                continue
            fids = [fid.item() for fid in data["fid"]]
            if any(fid not in saliency for fid in fids):
                continue
            if any(type(saliency[fid]) is not torch.Tensor for fid in fids):
                continue
            target = torch.cat([saliency[fid] for fid in fids]).cuda(
                non_blocking=True
            )
            hq_image = data["hq"].cuda(non_blocking=True)

            # inference
            with torch.no_grad():
                mask_slice = mask_generator(hq_image)

                # loss = 0
                # for idx, thresh in enumerate(thresholds):
                #     target = torch.cat(
                #         [saliency[thresh][fid].long().cuda() for fid in fids]
                #     )
                #     loss = loss + weight[idx] * get_loss(mask_slice, target, 1)
                loss = get_loss(mask_slice, target, thresh_list)

            if any(fid % 500 == 0 for fid in fids):
                if args.visualize:
                    maxid = np.argmax([fid % 500 == 0 for fid in fids]).item()
                    fid = fids[maxid]
                    image = T.ToPILImage()(data["hq"][maxid])
                    mask_slice = mask_slice[maxid : maxid + 1, :, :, :]
                    mask_slice = mask_slice.softmax(dim=1)[:, 1:2, :, :]
                    target = target[maxid : maxid + 1, :, :, :]
                    target = sum(
                        (target > thresh).float() for thresh in thresh_list
                    )
                    visualize_heat(
                        image,
                        mask_slice.detach().cpu(),
                        f"train/{args.path}/{fid}_cross.png",
                        args,
                    )
                    visualize_heat(
                        image,
                        target.cpu().detach(),
                        f"train/{args.path}/{fid}_saliency.png",
                        args,
                    )

            # optimization and logging
            if idx % 1 == 0:
                cross_writer.add_scalar(
                    Path(args.path).stem,
                    loss.item(),
                    idx
                    + iteration
                    * (len(training_set) + len(cross_validation_set))
                    + len(training_set),
                )
            cross_validation_losses.append(loss.item())

        mean_cross_validation_loss = (
            torch.tensor(cross_validation_losses).mean().item()
        )
        logger.info(
            "Average cross validation loss: %.3f", mean_cross_validation_loss
        )

        if mean_cross_validation_loss < mean_cross_validation_loss_before:
            mask_generator.save(args.path + ".best")
        mean_cross_validation_loss_before = min(
            mean_cross_validation_loss_before, mean_cross_validation_loss
        )

        # check if we need to reduce learning rate.
        scheduler.step(mean_cross_validation_loss)


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "-i",
    #     "--inputs",
    #     nargs="+",
    #     help="The video file name. The largest video file will be the ground truth.",
    #     required=True,
    # )
    # parser.add_argument('-s', '--source', type=str, help='The original video source.', required=True)
    # parser.add_argument('-g', '--ground_truth', type=str,
    #                     help='The ground truth videos.', required=True)
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="The path to store the generator parameters.",
        required=True,
    )
    parser.add_argument(
        "--init",
        type=str,
        help="The path to init the generator parameters.",
        required=True,
    )
    parser.add_argument(
        "--log", type=str, help="The logging file.", required=True,
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        help="The ground truth file.",
        required=True,
    )
    # parser.add_argument('-o', '--output', type=str,
    #                     help='The output name.', required=True)
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
    )
    parser.add_argument(
        "--maskgen_file",
        type=str,
        help="The file that defines the neural network.",
        required=True,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )
    parser.add_argument(
        "--saliency_threshold",
        type=float,
        help="The threshold to binarize the saliency.",
        default=0.5,
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of iterations for optimizing the mask.",
        default=500,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of iterations for optimizing the mask.",
        default=2,
    )
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, help="The learning rate.", default=1e-4
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="The gamma parameter for focal loss.",
        default=2,
    )
    parser.add_argument(
        "--visualize", type=bool, help="Visualize the heatmap.", default=False
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="The GPU id for distributed training",
    )

    args = parser.parse_args()

    main(args)
