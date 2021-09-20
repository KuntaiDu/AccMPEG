import math
from collections import namedtuple
from pdb import set_trace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, ModuleList, Sequential

GraphPath = namedtuple("GraphPath", ["s0", "name", "s1"])  #


class SSD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_net: nn.ModuleList,
        source_layer_indexes: List[int],
        extras: nn.ModuleList,
        classification_headers: nn.ModuleList,
        regression_headers: nn.ModuleList,
        is_test=False,
        config=None,
        device=None,
    ):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList(
            [
                t[1]
                for t in source_layer_indexes
                if isinstance(t, tuple) and not isinstance(t, GraphPath)
            ]
        )
        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError("This forward should not be called.")

        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index:end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[: path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1 :]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations,
                self.priors,
                self.config.center_variance,
                self.config.size_variance,
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):

        confidence = self.classification_headers[i](x)

        # confidence = confidence.permute(0, 2, 3, 1).contiguous()

        # confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        # we don't care about the locations
        # location = self.regression_headers[i](x)
        # location = location.permute(0, 2, 3, 1).contiguous()
        # location = location.view(location.size(0), -1, 4)

        return confidence

    # def init_from_base_net(self, model):
    #     self.base_net.load_state_dict(
    #         torch.load(model, map_location=lambda storage, loc: storage),
    #         strict=True,
    #     )
    #     self.source_layer_add_ons.apply(_xavier_init_)
    #     self.extras.apply(_xavier_init_)
    #     self.classification_headers.apply(_xavier_init_)
    #     self.regression_headers.apply(_xavier_init_)

    # def init_from_pretrained_ssd(self, model):
    #     state_dict = torch.load(
    #         model, map_location=lambda storage, loc: storage
    #     )
    #     state_dict = {
    #         k: v
    #         for k, v in state_dict.items()
    #         if not (
    #             k.startswith("classification_headers")
    #             or k.startswith("regression_headers")
    #         )
    #     }
    #     model_dict = self.state_dict()
    #     model_dict.update(state_dict)
    #     self.load_state_dict(model_dict)
    #     self.classification_headers.apply(_xavier_init_)
    #     self.regression_headers.apply(_xavier_init_)

    # def init(self):
    #     self.base_net.apply(_xavier_init_)
    #     self.source_layer_add_ons.apply(_xavier_init_)
    #     self.extras.apply(_xavier_init_)
    #     self.classification_headers.apply(_xavier_init_)
    #     self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(
            torch.load(model, map_location=lambda storage, loc: storage)
        )

    # def save(self, model_path):
    #     torch.save(self.state_dict(), model_path)


# class MatchPrior(object):
#     def __init__(
#         self, center_form_priors, center_variance, size_variance, iou_threshold
#     ):
#         self.center_form_priors = center_form_priors
#         self.corner_form_priors = box_utils.center_form_to_corner_form(
#             center_form_priors
#         )
#         self.center_variance = center_variance
#         self.size_variance = size_variance
#         self.iou_threshold = iou_threshold

#     def __call__(self, gt_boxes, gt_labels):
#         if type(gt_boxes) is np.ndarray:
#             gt_boxes = torch.from_numpy(gt_boxes)
#         if type(gt_labels) is np.ndarray:
#             gt_labels = torch.from_numpy(gt_labels)
#         boxes, labels = box_utils.assign_priors(
#             gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold
#         )
#         boxes = box_utils.corner_form_to_center_form(boxes)
#         locations = box_utils.convert_boxes_to_locations(
#             boxes,
#             self.center_form_priors,
#             self.center_variance,
#             self.size_variance,
#         )
#         return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False), ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False), ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio,
        use_batch_norm=True,
        onnx_compatible=False,
    ):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        n_class=1000,
        input_size=224,
        width_mult=1.0,
        dropout_ratio=0.2,
        use_batch_norm=True,
        onnx_compatible=False,
    ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        self.features = [
            conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)
        ]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            input_channel,
                            output_channel,
                            s,
                            expand_ratio=t,
                            use_batch_norm=use_batch_norm,
                            onnx_compatible=onnx_compatible,
                        )
                    )
                else:
                    self.features.append(
                        block(
                            input_channel,
                            output_channel,
                            1,
                            expand_ratio=t,
                            use_batch_norm=use_batch_norm,
                            onnx_compatible=onnx_compatible,
                        )
                    )
                input_channel = output_channel
        # building last several layers
        self.features.append(
            conv_1x1_bn(
                input_channel,
                self.last_channel,
                use_batch_norm=use_batch_norm,
                onnx_compatible=onnx_compatible,
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio), nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def SeperableConv2d(
    in_channels,
    out_channels,
    kernel_size=1,
    stride=1,
    padding=0,
    onnx_compatible=False,
):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
        ),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        ),
    )


def create_mobilenetv2_ssd_lite(
    num_classes,
    width_mult=1.0,
    use_batch_norm=True,
    onnx_compatible=False,
    is_test=False,
):
    base_net = MobileNetV2(
        width_mult=width_mult,
        use_batch_norm=use_batch_norm,
        onnx_compatible=onnx_compatible,
    ).features

    source_layer_indexes = [
        GraphPath(14, "conv", 3),
        19,
    ]
    extras = ModuleList(
        [
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25),
        ]
    )

    regression_headers = ModuleList(
        [
            SeperableConv2d(
                in_channels=round(576 * width_mult),
                out_channels=6 * 4,
                kernel_size=3,
                padding=1,
                onnx_compatible=False,
            ),
            SeperableConv2d(
                in_channels=1280,
                out_channels=6 * 4,
                kernel_size=3,
                padding=1,
                onnx_compatible=False,
            ),
            SeperableConv2d(
                in_channels=512,
                out_channels=6 * 4,
                kernel_size=3,
                padding=1,
                onnx_compatible=False,
            ),
            SeperableConv2d(
                in_channels=256,
                out_channels=6 * 4,
                kernel_size=3,
                padding=1,
                onnx_compatible=False,
            ),
            SeperableConv2d(
                in_channels=256,
                out_channels=6 * 4,
                kernel_size=3,
                padding=1,
                onnx_compatible=False,
            ),
            Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ]
    )

    classification_headers = ModuleList(
        [
            SeperableConv2d(
                in_channels=round(576 * width_mult),
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1,
            ),
            SeperableConv2d(
                in_channels=1280,
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1,
            ),
            SeperableConv2d(
                in_channels=512,
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1,
            ),
            SeperableConv2d(
                in_channels=256,
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1,
            ),
            SeperableConv2d(
                in_channels=256,
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1,
            ),
            Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ]
    )

    return SSD(
        num_classes,
        base_net,
        source_layer_indexes,
        extras,
        classification_headers,
        regression_headers,
        is_test=is_test,
        config=None,
    )


class FCN(nn.Module):
    def __init__(self, architecture="SSD"):
        super(FCN, self).__init__()

        self.model = create_mobilenetv2_ssd_lite(21)
        self.model.load("downloaded_weight/mb2-ssd-lite-mp-0_686.pth")
        self.architecture = architecture

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Batchnorm? don't need that cuz the inputs are already probabilities
        # self.process = nn.Sequential(
        #     nn.Conv2d(216, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 2, 3, padding=1),
        # )

        self.process = nn.Sequential(
            nn.BatchNorm2d(3700),
            nn.Conv2d(3700, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.BatchNorm2d(2),
        )

    """
        Deprecated forwarding code that generates heats based on feature rather than the confidence.
    """

    # def forward(self, x):

    #     start_layer_index = 0
    #     header_index = 0

    #     fpn = []

    #     for end_layer_index in self.model.source_layer_indexes:
    #         if isinstance(end_layer_index, GraphPath):
    #             path = end_layer_index
    #             end_layer_index = end_layer_index.s0
    #             added_layer = None
    #         elif isinstance(end_layer_index, tuple):
    #             added_layer = end_layer_index[1]
    #             end_layer_index = end_layer_index[0]
    #             path = None
    #         else:
    #             added_layer = None
    #             path = None
    #         for layer in self.model.base_net[start_layer_index:end_layer_index]:
    #             x = layer(x)
    #             fpn.append(x)
    #         if added_layer:
    #             y = added_layer(x)
    #         else:
    #             y = x
    #         if path:
    #             sub = getattr(self.model.base_net[end_layer_index], path.name)
    #             for layer in sub[: path.s1]:
    #                 x = layer(x)
    #                 fpn.append(x)
    #             y = x
    #             for layer in sub[path.s1 :]:
    #                 x = layer(x)
    #                 fpn.append(x)
    #             end_layer_index += 1
    #         start_layer_index = end_layer_index
    #         confidence, location = self.model.compute_header(header_index, y)
    #         header_index += 1

    #     for layer in self.model.base_net[end_layer_index:]:
    #         x = layer(x)
    #         fpn.append(x)

    #     fpn = [fpn[10], fpn[13], fpn[16], fpn[19], fpn[23], fpn[24], fpn[25]]
    #     fpn = [F.interpolate(i, (45, 80), mode="bilinear") for i in fpn]
    #     features = torch.cat(fpn, dim=1)

    #     return self.process(features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():

            confidences = []
            fpn = []
            start_layer_index = 0
            header_index = 0
            for end_layer_index in self.model.source_layer_indexes:
                if isinstance(end_layer_index, GraphPath):
                    path = end_layer_index
                    end_layer_index = end_layer_index.s0
                    added_layer = None
                elif isinstance(end_layer_index, tuple):
                    added_layer = end_layer_index[1]
                    end_layer_index = end_layer_index[0]
                    path = None
                else:
                    added_layer = None
                    path = None
                for layer in self.model.base_net[
                    start_layer_index:end_layer_index
                ]:
                    x = layer(x)
                if added_layer:
                    y = added_layer(x)
                else:
                    y = x
                if path:
                    sub = getattr(
                        self.model.base_net[end_layer_index], path.name
                    )
                    for layer in sub[: path.s1]:
                        x = layer(x)
                    y = x
                    for layer in sub[path.s1 :]:
                        x = layer(x)
                    end_layer_index += 1
                start_layer_index = end_layer_index
                confidence = self.model.compute_header(header_index, y)
                header_index += 1
                confidences.append(confidence)
                fpn.append(y)

            for layer in self.model.base_net[end_layer_index:]:
                x = layer(x)

            for layer in self.model.extras:
                x = layer(x)
                confidence = self.model.compute_header(header_index, x)
                fpn.append(x)
                header_index += 1
                confidences.append(confidence)

            confidences = [F.interpolate(i, (45, 80)) for i in confidences]
            confidences = torch.cat(confidences, 1)
            fpn = [F.interpolate(i, (45, 80)) for i in fpn]
            fpn = torch.cat(fpn, 1)

            confidences = torch.cat([confidences, fpn], dim=1)
            # confidences = confidences.split(21, dim=1)
            # confidences = [i.softmax(dim=1) for i in confidences]
            # # 0: background
            # # 6: bus
            # # 7: car
            # # 14: motorbike
            # # 15: person
            # # 19: train
            # confidences = [i[:, [0, 6, 7, 14, 15, 19], :, :] for i in confidences]

            # confidences = torch.cat(confidences, dim=1)

        return self.process(confidences)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
