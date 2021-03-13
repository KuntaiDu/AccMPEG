import logging
from pdb import set_trace

import detectron2
import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import pairwise_iou
from detectron2.structures.keypoints import Keypoints
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import Visualizer
from PIL import Image

from .dnn import DNN

panoptic_segmentation_labels = [
    "things",
    "banner",
    "blanket",
    "bridge",
    "cardboard",
    "counter",
    "curtain",
    "door-stuff",
    "floor-wood",
    "flower",
    "fruit",
    "gravel",
    "house",
    "light",
    "mirror-stuff",
    "net",
    "pillow",
    "platform",
    "playingfield",
    "railroad",
    "river",
    "road",
    "roof",
    "sand",
    "sea",
    "shelf",
    "snow",
    "stairs",
    "tent",
    "towel",
    "wall-brick",
    "wall-stone",
    "wall-tile",
    "wall-wood",
    "water",
    "window-blind",
    "window",
    "tree",
    "fence",
    "ceiling",
    "sky",
    "cabinet",
    "table",
    "floor",
    "pavement",
    "mountain",
    "grass",
    "dirt",
    "paper",
    "food",
    "building",
    "rock",
    "wall",
    "rug",
]


class COCO_Model(DNN):
    def __init__(self, name):

        self.name = name

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(name))
        self.cfg.DOWNLOAD_CACHE = "/data2/kuntai/torch/detectron2/"

        # # to make it run on cpu, just for measurement purpose.
        # self.cfg.MODEL.DEVICE='cpu'
        # filter out those regions that has confidence score < 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(name)
        self.predictor = None

        self.logger = logging.getLogger(self.name)
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.class_ids = [0, 1, 2, 3, 5, 6, 7]

        self.keys = [
            "scores",
            "pred_classes",
            "pred_boxes",
            "pred_masks",
            "pred_keypoints",
            "proposals",
        ]

    # def cpu(self):

    #     self.predictor.model.cpu()
    #     self.logger.info(f"Place %s on CPU.", self.name)

    # def cuda(self):

    #     self.model.cuda()
    #     self.logger.info(f"Place %s on GPU.", self.name)

    def preprocess_image(self, image):

        assert (
            len(image.shape) == 4 and image.shape[0] == 1
        ), "Only deal with one image to avoid GPU memory overflow."

        h, w = image.shape[2], image.shape[3]

        if self.cfg.INPUT.FORMAT == "BGR":
            image = image[:, [2, 1, 0], :, :]
        else:
            assert self.cfg.INPUT.FORMAT == "RGB"

        image = image * 255
        transform = self.predictor.aug.get_transform(image[0].permute(1, 2, 0))
        image = F.interpolate(image, (transform.new_h, transform.new_w))

        return image, h, w, transform

    def inference(self, image, detach=False, grad=False):

        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.eval()

        image, h, w, _ = self.preprocess_image(image)

        with torch.enable_grad() if grad else torch.no_grad():
            ret = self.predictor.model(
                [{"image": image[0], "height": h, "width": w}]
            )[0]

        if detach:
            for key in ret:
                # this will also store the region proposal info
                ret[key] = ret[key].to("cpu")
        return ret

    def region_proposal(self, image, detach=False, grad=False):

        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.eval()

        image, h, w, _ = self.preprocess_image(image)

        with torch.enable_grad() if grad else torch.no_grad():
            model = self.predictor.model
            x = [{"image": image[0], "height": h, "width": w}]
            images = model.preprocess_image(x)
            features = model.backbone(images.tensor)
            proposals, logits = model.proposal_generator(images, features, None)

        ret = proposals[0]

        if detach:
            ret = ret.to("cpu")
        return ret

    def filter_result(self, result, args, gt=False, confidence_check=True):

        scores = result["instances"].scores
        class_ids = result["instances"].pred_classes

        inds = scores < 0
        for i in self.class_ids:
            inds = inds | (class_ids == i)

        if confidence_check:
            if gt:
                inds = inds & (scores > args.gt_confidence_threshold)
            else:
                inds = inds & (scores > args.confidence_threshold)

        result["instances"] = result["instances"][inds]

        return result

    def visualize(self, image, result, args):
        # set_trace()
        result = self.filter_result(result, args)
        v = Visualizer(
            image, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1
        )
        out = v.draw_instance_predictions(result["instances"])
        return Image.fromarray(out.get_image(), "RGB")

    def calc_loss(self, image, result, args):

        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model.train()

        result = self.filter_result(result, args)

        if result["instances"].has("pred_classes"):
            result["instances"].gt_classes = result["instances"].pred_classes
        if result["instances"].has("pred_boxes"):
            result["instances"].gt_boxes = result["instances"].pred_boxes
        if result["instances"].has("pred_masks"):
            result["instances"].gt_masks = result["instances"].pred_masks
        if result["instances"].has("pred_keypoints"):
            result["instances"].gt_keypoints = Keypoints(
                result["instances"].pred_keypoints
            )

        # convert result to target
        image, h, w, _ = self.preprocess_image(image)

        # Detectron2 models must be wrapped with EventStorage to run normally
        with EventStorage() as storage:

            ret = self.predictor.model(
                [
                    {
                        "image": image[0],
                        "height": h,
                        "width": w,
                        "instances": result["instances"],
                    }
                ]
            )

        return sum(ret.values())

    def calc_accuracy(self, result_dict, gt_dict, args):

        if "Detection" in self.name:
            return self.calc_accuracy_detection(result_dict, gt_dict, args)
        elif "keypoint" in self.name:
            return self.calc_accuracy_keypoint(result_dict, gt_dict, args)

    def calc_accuracy_loss(self, image, gt, args):

        result = self.inference(image, detach=False, grad=True)

        if "Detection" in self.name:
            return self.calc_accuracy_loss_detection(result, gt, args)
        else:
            raise NotImplementedError()

    def calc_accuracy_detection(self, result_dict, gt_dict, args):

        assert (
            result_dict.keys() == gt_dict.keys()
        ), "Result and ground truth must contain the same number of frames."

        f1s = []
        prs = []
        res = []
        tps = []
        fps = []
        fns = []

        for fid in result_dict.keys():
            result = result_dict[fid]
            gt = gt_dict[fid]

            result = self.filter_result(result, args, False)
            gt = self.filter_result(gt, args, True)

            result = result["instances"]
            gt = gt["instances"]

            if len(result) == 0 or len(gt) == 0:
                if len(result) == 0 and len(gt) == 0:
                    f1s.append(1.0)
                    prs.append(1.0)
                    res.append(1.0)
                else:
                    f1s.append(0.0)
                    prs.append(0.0)
                    res.append(0.0)

            IoU = pairwise_iou(result.pred_boxes, gt.pred_boxes)

            for i in range(len(result)):
                for j in range(len(gt)):
                    if result.pred_classes[i] != gt.pred_classes[j]:
                        IoU[i, j] = 0

            tp = 0

            for i in range(len(gt)):
                if sum(IoU[:, i] > args.iou_threshold):
                    tp += 1
            fn = len(gt) - tp
            fp = len(result) - tp
            fp = max(fp, 0)

            if 2 * tp + fp + fn == 0:
                f1 = 1.0
            else:
                f1 = 2 * tp / (2 * tp + fp + fn)
            if tp + fp == 0:
                pr = 1.0
            else:
                pr = tp / (tp + fp)
            if tp + fn == 0:
                re = 1.0
            else:
                re = tp / (tp + fn)

            f1s.append(f1)
            prs.append(pr)
            res.append(re)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

        sum_tp = sum(tps)
        sum_fp = sum(fps)
        sum_fn = sum(fns)

        return {
            "f1": torch.tensor(f1s).mean().item(),
            "pr": torch.tensor(prs).mean().item(),
            "re": torch.tensor(res).mean().item(),
            "tp": torch.tensor(tps).sum().item(),
            "fp": torch.tensor(fps).sum().item(),
            "fn": torch.tensor(fns).sum().item(),
            "sum_f1": (2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn))
            # "f1s": f1s,
            # "prs": prs,
            # "res": res,
            # "tps": tps,
            # "fns": fns,
            # "fps": fps,
        }

    def calc_accuracy_keypoint(self, result_dict, gt_dict, args):
        f1s = []
        # prs = []
        # res = []
        # tps = []
        # fps = []
        # fns = []
        for fid in result_dict.keys():
            result = result_dict[fid]["instances"].get_fields()
            gt = gt_dict[fid]["instances"].get_fields()
            if len(gt["scores"]) == 0 and len(result["scores"]) == 0:
                # prs.append(0.0)
                # res.append(0.0)
                f1s.append(1.0)
                # tps.append(0.0)
                # fps.append(0.0)
                # fns.append(0.0)
            elif len(result["scores"]) == 0 or len(gt["scores"]) == 0:
                # prs.append(0.0)
                # res.append(0.0)
                f1s.append(0.0)
                # tps.append(0.0)
                # fps.append(0.0)
                # fns.append(0.0)
            else:
                video_ind_res = result["scores"] == torch.max(result["scores"])
                kpts_res = result["pred_keypoints"][video_ind_res]
                video_ind_gt = gt["scores"] == torch.max(gt["scores"])
                kpts_gt = gt["pred_keypoints"][video_ind_gt]

                try:
                    acc = kpts_res - kpts_gt
                except:
                    import pdb

                    pdb.set_trace()
                    print("shouldnt happen")

                gt_boxes = gt["pred_boxes"][video_ind_gt].tensor
                kpt_thresh = float(args.dist_thresh)

                acc = acc[0]
                acc = torch.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2)
                # acc[acc < kpt_thresh * kpt_thresh] = 0
                for i in range(len(acc)):
                    max_dim = max(
                        (gt_boxes[i // 17][2] - gt_boxes[i // 17][0]),
                        (gt_boxes[i // 17][3] - gt_boxes[i // 17][1]),
                    )
                    if acc[i] < (max_dim * kpt_thresh) ** 2:
                        acc[i] = 0

                accuracy = 1 - (len(acc.nonzero()) / acc.numel())
                # prs.append(0.0)
                # res.append(0.0)
                f1s.append(accuracy)
                # tps.append(0.0)
                # fps.append(0.0)
                # fns.append(0.0)

        return {
            "f1": torch.tensor(f1s).mean().item(),
            # "pr": torch.tensor(prs).mean().item(),
            # "re": torch.tensor(res).mean().item(),
            # "tp": torch.tensor(tps).sum().item(),
            # "fp": torch.tensor(fps).sum().item(),
            # "fn": torch.tensor(fns).sum().item(),
            # "f1s": f1s,
            # "prs": prs,
            # "res": res,
            # "tps": tps,
            # "fns": fns,
            # "fps": fps,
        }

    def calc_accuracy_loss_detection(self, result, gt, args):

        from detectron2.structures.boxes import pairwise_iou

        gt = self.filter_result(gt, args, True)
        result = self.filter_result(result, args, False, confidence_check=False)

        result = result["instances"]
        gt = gt["instances"].to("cuda")

        if len(result) == 0 or len(gt) == 0:
            if len(result) == 0 and len(gt) == 0:
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        IoU = pairwise_iou(result.pred_boxes, gt.pred_boxes)

        for i in range(len(result)):
            for j in range(len(gt)):
                if result.pred_classes[i] != gt.pred_classes[j]:
                    IoU[i, j] = 0

        tp = 0

        def f(x):
            a = args.alpha
            x = x - args.confidence_threshold
            # x == -a: 0, x == a: 1
            res = (x + a) / (2 * a)
            res = max(res, 0)
            res = min(res, 1)
            return res

        for j in range(len(gt)):
            tp_delta = 0
            for i in range(len(result)):
                if IoU[i, j] > args.iou_threshold:
                    # IoU threshold will be hard threshold
                    tp_delta += max(tp_delta, f(result.scores[i]))
            tp = tp + tp_delta

        fn = len(gt) - tp
        fp = sum([f(result.scores[i]) for i in range(len(result))]) - tp
        fp = max(fp, 0)

        return -2 * tp / (2 * tp + fp + fn)

    def get_undetected_ground_truth_index(self, result, gt, args):

        gt = self.filter_result(gt, args, True)
        result = self.filter_result(result, args, False)

        result = result["instances"]
        gt = gt["instances"]

        IoU = pairwise_iou(result.pred_boxes, gt.pred_boxes)
        for i in range(len(result)):
            for j in range(len(gt)):
                if result.pred_classes[i] != gt.pred_classes[j]:
                    IoU[i, j] = 0

        return (IoU > args.iou_threshold).sum(dim=0) == 0
