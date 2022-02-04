import argparse
import io
import os
from typing import Dict, List, Tuple

import detectron2.data.transforms as T
import numpy as np
import torch
import torch.onnx
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.export import (
    Caffe2Tracer,
    TracingAdapter,
    add_export_config,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from munch import Munch
from torch import Tensor, nn

from maskgen.SSD.accmpegmodel import FCN

"""
    Export AccMPEG
"""

# model = FCN()
# model.load(
#     "maskgen_pths/COCO_detection_FPN_SSD_withconfidence_allclasses_new_unfreezebackbone_withoutclasscheck.pth.best"
# )
# model.eval()


# x = torch.randn([1, 3, 720, 1280])
# out = model(x)
# print(out.shape)

# torch.onnx.export(
#     model,
#     x,
#     "measurements/onnx/accmpeg.onnx",
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=["input"],
#     output_names=["output"],
#     # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
# )


"""
    Export MobileNet SSD for Vigil
    Code edited from https://github.com/facebookresearch/detectron2/blob/main/tools/deploy/export_model.py
"""


# MobileNet SSD
# SSD = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd")
# SSD.eval()
# x = torch.randn([1, 3, 300, 300])
# out = SSD(x)


# torch.onnx.export(
#     SSD,
#     x,
#     "measurements/onnx/vigl_SSD.onnx",
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=["input"],
#     output_names=["output"],
#     # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
# )

args = Munch()


def setup_cfg(args):
    cfg = get_cfg()
    cfg = add_export_config(cfg)
    add_pointrend_config(cfg)
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        )
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.DOWNLOAD_CACHE = "/data2/kuntai/torch/detectron2/"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    cfg.freeze()
    return cfg


def export_caffe2_tracing(cfg, torch_model, inputs):
    tracer = Caffe2Tracer(cfg, torch_model, inputs)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(
            os.path.join(args.output, "model.svg"), inputs=inputs
        )
        return caffe2_model
    elif args.format == "onnx":
        import onnx

        onnx_model = tracer.export_onnx()
        onnx.save(
            onnx_model, os.path.join(args.output, "FasterRCNN-ResNet101.onnx")
        )
    elif args.format == "torchscript":
        ts_model = tracer.export_torchscript()
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)


def get_sample_inputs(args):

    if args.sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(
            args.sample_image, format=cfg.INPUT.FORMAT
        )
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


args.format = "onnx"
args.export_method = "caffe2_tracing"
args.sample_image = "videos/dashcamcropped_1/0000000000.png"
args.run_eval = False
args.output = "./measurements/onnx/"


logger = setup_logger()
logger.info("Command line arguments: " + str(args))
PathManager.mkdirs(args.output)
# Disable respecialization on new shapes. Otherwise --run-eval will be slow
torch._C._jit_set_bailout_depth(1)

cfg = setup_cfg(args)

# create a torch model
torch_model = build_model(cfg)
DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
torch_model.eval()

# get sample data
sample_inputs = get_sample_inputs(args)

exported_model = export_caffe2_tracing(cfg, torch_model, sample_inputs)
