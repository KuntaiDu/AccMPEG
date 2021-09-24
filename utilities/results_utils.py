import pickle
from pathlib import Path

import torch

from utilities.bbox_utils import jaccard


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


# def merge_results(gt, video, application, args):

#     # merge two bounding boxes into a larger one if they
#     assert sorted(gt.keys()) == sorted(video.keys())

#     # the return results
#     ret = {}

#     for fid in gt:
#         gt_ind, gt_scores, gt_boxes, gt_labels = application.filter_results(
#             gt[fid], args.confidence_threshold
#         )
#         assert len(gt_ind) == len(gt_labels)
#         v_ind, v_scores, v_boxes, v_labels = application.filter_results(
#             video[fid], args.confidence_threshold
#         )

#         # get IoU
#         IoU = jaccard(v_boxes, gt_boxes)

#         # eliminate those IoU values if the label is wrong
#         fat_v_labels = v_labels[:, None].repeat(1, len(gt_labels))
#         fat_gt_labels = gt_labels[None, :].repeat(len(v_labels), 1)
#         IoU[fat_v_labels != fat_gt_labels] = 0

#         # eliminate those bounding boxes from video that has high IoU w/ ground truth
#         for i in range(len(v_labels)):
#             if any(IoU[i, :] > args.iou_threshold):
#                 v_ind[i] = False
#             else:
#                 v_ind[i] = True

#         ret_scores = torch.cat([gt_scores, v_scores[v_ind]], dim=0)
#         ret_boxes = torch.cat([gt_boxes, v_boxes[v_ind, :]], dim=0)
#         ret_labels = torch.cat([gt_labels, v_labels[v_ind]], dim=0)

#         ret[fid] = {"boxes": ret_boxes, "scores": ret_scores, "labels": ret_labels}

#     return ret


def merge_results(results, app, args):

    return results[0]
    #import enlighten
    #import networkx as nx

    #for i in range(1, len(results)):
    #    assert len(results[i - 1].keys()) == len(
    #        results[i].keys()
    #    ), "Results must contain same amount of frames"

    #new_results = {}

    #progress_bar = enlighten.get_manager().counter(
    #    total=len(results[0].keys()), desc=f"Generate ground truth", unit="frames"
    #)

    #for fid in results[0].keys():

    #    progress_bar.update()

    #    results_fid = [result[fid] for result in results]
    #    boxes = []
    #    boxes_tensor = []

    #    for result in results_fid:

    #        _, score, box, label = app.filter_results(result, args.confidence_threshold)
    #        assert len(score) == len(box) == len(label)

    #        for i in range(len(score)):
    #            boxes.append(
    #                {
    #                    "score": score[i : i + 1],
    #                    "box": box[i : i + 1, :],
    #                    "label": label[i : i + 1],
    #                }
    #            )
    #            boxes_tensor.append(box[i : i + 1, :])

    #    g = nx.Graph()

    #    for idx, box in enumerate(boxes):
    #        g.add_node(idx)

    #    if boxes == []:
    #        new_results[fid] = {
    #            "scores": torch.zeros([0]),
    #            "boxes": torch.zeros([0, 4]),
    #            "labels": torch.zeros([0]),
    #        }
    #        continue

    #    boxes_tensor = torch.cat(boxes_tensor)
    #    IoU = jaccard(boxes_tensor, boxes_tensor)

    #    for index in (IoU > args.iou_threshold).nonzero():

    #        if boxes[index[0]]["label"] == boxes[index[1]]["label"]:
    #            g.add_edge(index[0].item(), index[1].item())

    #    # generate new boxes
    #    new_boxes = []
    #    new_scores = []
    #    new_labels = []
    #    for comp in nx.connected_components(g):

    #        if len(comp) <= 2:
    #            continue

    #        new_boxes.append(
    #            torch.cat([boxes[node]["box"] for node in comp]).mean(
    #                dim=0, keepdim=True
    #            )
    #        )
    #        new_scores.append(
    #            torch.cat([boxes[node]["score"] for node in comp]).mean(
    #                dim=0, keepdim=True
    #            )
    #        )
    #        new_labels.append(boxes[list(comp)[0]]["label"])

    #    if new_boxes == []:
    #        new_results[fid] = {
    #            "scores": torch.zeros([0]),
    #            "boxes": torch.zeros([0, 4]),
    #            "labels": torch.zeros([0]),
    #        }
    #    else:
    #        new_results[fid] = {
    #            "scores": torch.cat(new_scores),
    #            "boxes": torch.cat(new_boxes),
    #            "labels": torch.cat(new_labels),
    #        }

    #return new_results


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


def clean_results(gt, videos, app, args):

    raise NotImplementedError("Plan to implement it if the sanity check works.")
