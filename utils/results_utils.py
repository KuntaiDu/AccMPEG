import pickle
from pathlib import Path

import torch

from utils.bbox_utils import jaccard


def write_results(video_name, app_name, results, logger):

    logger.info(
        f"Writing inference results of application {app_name} on video {video_name}."
    )
    results_file = Path(f"results/{app_name}/{video_name}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "wb") as f:
        pickle.dump(results, f)


def read_results(video_name, app_name, logger):

    logger.info(
        f"Reading inference results of application {app_name} on video {video_name}."
    )
    results_file = Path(f"results/{app_name}/{video_name}")
    with open(results_file, "rb") as f:
        return pickle.load(f)


def merge_results(gt, video, application, args):

    # merge two bounding boxes into a larger one if they
    assert sorted(gt.keys()) == sorted(video.keys())

    # the return results
    ret = {}

    for fid in gt:
        gt_ind, gt_scores, gt_boxes, gt_labels = application.filter_results(
            gt[fid], args.confidence_threshold
        )
        assert len(gt_ind) == len(gt_labels)
        v_ind, v_scores, v_boxes, v_labels = application.filter_results(
            video[fid], args.confidence_threshold
        )

        # get IoU
        IoU = jaccard(v_boxes, gt_boxes)

        # eliminate those IoU values if the label is wrong
        fat_v_labels = v_labels[:, None].repeat(1, len(gt_labels))
        fat_gt_labels = gt_labels[None, :].repeat(len(v_labels), 1)
        IoU[fat_v_labels != fat_gt_labels] = 0

        # eliminate those bounding boxes from video that has high IoU w/ ground truth
        for i in range(len(v_labels)):
            if any(IoU[i, :] > args.iou_threshold):
                v_ind[i] = False
            else:
                v_ind[i] = True

        ret_scores = torch.cat([gt_scores, v_scores[v_ind]], dim=0)
        ret_boxes = torch.cat([gt_boxes, v_boxes[v_ind, :]], dim=0)
        ret_labels = torch.cat([gt_labels, v_labels[v_ind]], dim=0)

        ret[fid] = {"boxes": ret_boxes, "scores": ret_scores, "labels": ret_labels}

    return ret


def read_ground_truth(file_name, logger):

    ground_truths = {}
    logger.info("Load ground truth from %s", file_name)

    with open(file_name, "rb") as f:
        try:
            while True:
                ground_truth = pickle.load(f)
                ground_truths.update(ground_truth)
        except EOFError:
            pass

    logger.info("Ground truth loaded.")

    return ground_truths


def read_ground_truth_mask(file_name, logger):

    logger.info("Load ground truth mask from %s", file_name)

    with open(file_name, "rb") as f:
        return pickle.load(f)
