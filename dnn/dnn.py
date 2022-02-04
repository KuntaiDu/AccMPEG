from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.visualizer import Visualizer
from PIL import Image


class DNN(ABC):
    # @abstractmethod
    # def cpu(self):
    #     pass

    # @abstractmethod
    # def cuda(self):
    #     pass

    @abstractmethod
    def inference(self, video, requires_grad):
        pass

    def filter_result(
        self,
        result,
        args,
        gt=False,
        confidence_check=True,
        require_deepcopy=False,
        class_check=True,
    ):

        if require_deepcopy:
            result = deepcopy(result)

        scores = result["instances"].scores
        class_ids = result["instances"].pred_classes

        inds = scores < 0
        if class_check:
            for i in self.class_ids:
                inds = inds | (class_ids == i)
        else:
            inds = scores > -1

        # if confidence_check:
        #    if gt:
        #        inds = inds & (scores > args.gt_confidence_threshold)
        #    else:
        #        inds = inds & (scores > args.confidence_threshold)

        if confidence_check:
            if gt:
                inds = inds & (scores > args.gt_confidence_threshold)
            else:
                inds = inds & (scores > args.confidence_threshold)

        # result["instances"] = result["instances"][inds]

        return {"instances": result["instances"][inds]}

    def visualize(self, image, result):
        # set_trace()
        # result = self.filter_result(result, args, gt=gt)
        v = Visualizer(image, MetadataCatalog.get("coco_2017_train"), scale=1)
        out = v.draw_instance_predictions(result["instances"])
        return Image.fromarray(out.get_image(), "RGB")

    def calc_accuracy(self, result_dict, gt_dict, args):

        if self.type == "Detection":
            return self.calc_accuracy_detection(result_dict, gt_dict, args)
        elif self.type == "Keypoint":
            return self.calc_accuracy_keypoint(result_dict, gt_dict, args)

    # def calc_accuracy_loss(self, image, gt, args):

    #     result = self.inference(image, detach=False, grad=True)

    #     if "Detection" in self.name:
    #         return self.calc_accuracy_loss_detection(result, gt, args)
    #     else:
    #         raise NotImplementedError()

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

        if 2 * sum_tp + sum_fp + sum_fn == 0:
            sum_f1 = 1.0
        else:
            sum_f1 = 2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn)

        return {
            "f1": torch.tensor(f1s).mean().item(),
            "pr": torch.tensor(prs).mean().item(),
            "re": torch.tensor(res).mean().item(),
            "tp": torch.tensor(tps).sum().item(),
            "fp": torch.tensor(fps).sum().item(),
            "fn": torch.tensor(fns).sum().item(),
            "sum_f1": sum_f1
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

    # def calc_accuracy_loss_detection(self, result, gt, args):
    #     # from detectron2.structures.boxes import pairwise_iou
    #     gt = self.filter_result(gt, args, True)
    #     result = self.filter_result(result, args, False, confidence_check=False)

    #     result = result["instances"]
    #     gt = gt["instances"].to("cuda")

    #     if len(result) == 0 or len(gt) == 0:
    #         if len(result) == 0 and len(gt) == 0:
    #             return torch.tensor(1)
    #         else:
    #             return torch.tensor(0)

    #     IoU = pairwise_iou(result.pred_boxes, gt.pred_boxes)

    #     for i in range(len(result)):
    #         for j in range(len(gt)):
    #             if result.pred_classes[i] != gt.pred_classes[j]:
    #                 IoU[i, j] = 0

    #     tp = 0

    #     def f(x):
    #         a = args.alpha
    #         x = x - args.confidence_threshold
    #         # x == -a: 0, x == a: 1
    #         res = (x + a) / (2 * a)
    #         res = max(res, 0)
    #         res = min(res, 1)
    #         return res

    #     for j in range(len(gt)):
    #         tp_delta = 0
    #         for i in range(len(result)):
    #             if IoU[i, j] > args.iou_threshold:
    #                 # IoU threshold will be hard threshold
    #                 tp_delta += max(tp_delta, f(result.scores[i]))
    #         tp = tp + tp_delta

    #     fn = len(gt) - tp
    #     fp = sum([f(result.scores[i]) for i in range(len(result))]) - tp
    #     fp = max(fp, 0)

    #     return -2 * tp / (2 * tp + fp + fn)

    def get_undetected_ground_truth_index(self, result, gt, args):

        if self.type == "Segmentation":
            raise NotImplementedError

        gt = deepcopy(gt)
        result = deepcopy(result)

        gt = self.filter_result(gt, args, gt=True)
        result = self.filter_result(result, args, gt=False)

        result = result["instances"]
        gt = gt["instances"]

        IoU = pairwise_iou(result.pred_boxes, gt.pred_boxes)
        for i in range(len(result)):
            for j in range(len(gt)):
                if result.pred_classes[i] != gt.pred_classes[j]:
                    IoU[i, j] = 0

        return (
            (IoU > args.iou_threshold).sum(dim=0) == 0,
            (IoU > args.iou_threshold).sum(dim=1) == 0,
            gt,
            result,
        )

    def aggregate_inference_results(self, results, args):

        if self.type == "Detection":
            return self.aggregate_inference_results_detection(results, args)
        else:
            raise NotImplementedError

    def aggregate_inference_results_detection(self, results, args):

        base = results[0]["instances"]

        scores = [base.scores]

        for result in results[1:]:

            result = deepcopy(result["instances"])

            if len(base) == 0 or len(result) == 0:
                continue

            IoU = pairwise_iou(result.pred_boxes, base.pred_boxes)

            for i in range(len(result)):
                for j in range(len(base)):
                    if result.pred_classes[i] != base.pred_classes[j]:
                        IoU[i, j] = 0

            val, idx = IoU.max(dim=0)

            # clear those scores where IoU is way too small
            result[idx].scores[val < args.iou_threshold] = 0.0
            scores.append(result[idx].scores)

        scores = torch.cat([i.unsqueeze(0) for i in scores], dim=0)

        base.pred_scores = torch.tensor(scores).mean(dim=0)
        base.pred_std = torch.tensor(scores).std(dim=0)

        print(base.pred_std)

        return {"instances": base}
