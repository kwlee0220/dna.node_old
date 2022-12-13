"""
Implements the Generalized R-CNN for SiamMOT
"""
import torch
from torch import nn
import numpy as np

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

from gluoncv.utils.viz import plot_bbox

from ..roi_heads.track_heads import build_track_heads
from ..box_head.siam_fast_rcnn import fast_rcnn_inference

@META_ARCH_REGISTRY.register()
class MOT_RCNN(nn.Module):
    def __init__(self, cfg):
        super(MOT_RCNN, self).__init__()

        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.track_heads = build_track_heads(cfg, self.roi_heads)
        self.track_memory = None

    def flush_memory(self, cache=None):
        self.track_memory = cache

    def reset_siammot_status(self):
        self.flush_memory()
        self.track_heads.reset_roi_status()

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs):
        # if self.training and targets is None:
        #     raise ValueError("In training mode, targets should be passed")
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            targets = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            targets = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, targets)
        if self.training:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        detections, detector_losses = self.roi_heads(images, features, proposals, targets)

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            track_losses = {}

        if self.track_heads:
            x, result, track_losses = self.track_heads(images,
                                                       features,
                                                       proposals,
                                                       detections=detections,
                                                       targets=targets,
                                                       track_memory=self.track_memory)
            if not self.training:
                 self.flush_memory(cache=x)

        else:
            result = detections

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            if self.track_heads:
                losses.update(track_losses)
            return losses
        else:
             return self._postprocess(result, batched_inputs, images.image_sizes)

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            #r.pred_boxes.tensor = BoxMode.convert(r.pred_boxes.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) ## XYXY to XYWH
            processed_results.append({"instances": r})
        return processed_results

def build_siammot(cfg):
    siammot = MOT_RCNN(cfg)
    return siammot
