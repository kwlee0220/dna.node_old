from torch import nn
import torch.nn.functional as F

from detectron2.layers import Conv2d
from detectron2.modeling.poolers import ROIPooler


class EMMFeatureExtractor(nn.Module):
    """
    Feature extraction for template and search region is different in this case
    """

    def __init__(self, cfg):
        super(EMMFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.TRACK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.TRACK_HEAD.POOLER_SCALES
        self.feature_len = len(scales)
        sampling_ratio = cfg.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO
        r = cfg.MODEL.TRACK_HEAD.SEARCH_REGION

        pooler_z = ROIPooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2")

        pooler_x = ROIPooler(
            output_size=(int(resolution*r), int(resolution*r)),
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2")

        self.pooler_x = pooler_x
        self.pooler_z = pooler_z

    def forward(self, x, proposals, sr=None):
        if sr is not None:
            if sr[0].has("bbox"):
                boxlist = [p.bbox for p in sr]
            else:
                boxlist = [p.pred_boxes for p in sr]
            x = self.pooler_x(list(x[:self.feature_len]), boxlist)
        else:
            if proposals[0].has("bbox"):
                boxlist = [p.bbox for p in proposals]
            else:
                boxlist = [p.pred_boxes for p in proposals]
            x = self.pooler_z(list(x[:self.feature_len]), boxlist)
        return x

class EMMPredictor(nn.Module):
    def __init__(self, cfg):
        super(EMMPredictor, self).__init__()

        in_channels = cfg.MODEL.TRACK_HEAD.PREDICTOR_CHANNELS

        self.cls_tower = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=in_channels, out_channels=in_channels, bias=False, padding=1),
            nn.GroupNorm(32, in_channels, affine=True),
            nn.ReLU(inplace=True)
        )

        self.reg_tower = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=in_channels, out_channels=in_channels, bias=False, padding=1),
            nn.GroupNorm(32, in_channels, affine=True),
            nn.ReLU(inplace=True)
        )

        self.cls = Conv2d(kernel_size=3, in_channels=in_channels, out_channels=2, padding=1)
        self.center = Conv2d(kernel_size=3, in_channels=in_channels, out_channels=1, padding=1)
        self.reg = Conv2d(kernel_size=3, in_channels=in_channels, out_channels=4, padding=1)

    def forward(self, x):
        cls_x = self.cls_tower(x)
        reg_x = self.reg_tower(x)
        cls_logits = self.cls(cls_x)
        center_logits = self.center(cls_x)
        reg_logits = F.relu(self.reg(reg_x))

        return cls_logits, center_logits, reg_logits