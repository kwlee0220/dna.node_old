import torch
from torch.nn import ModuleList

from ..track_head.track_head import build_track_head
from ..track_head.track_utils import build_track_utils
from ..track_head.track_solver import builder_tracker_solver
from ..box_head.siam_fast_rcnn import fast_rcnn_inference

from detectron2.structures import Instances, Boxes, pairwise_iou

class CombinedTrackHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """
    def __init__(self, cfg, head, box):
        super(CombinedTrackHeads, self).__init__(head)
        self.cfg = cfg.clone()
        self.device = cfg.MODEL.DEVICE

        self.max_dormant_frames = cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES
        self.track_thresh = cfg.MODEL.TRACK_HEAD.TRACK_THRESH
        self.start_thresh = cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH
        self.resume_track_thresh = cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH
        if not isinstance(box.box_head, ModuleList):
            self.box_head = [box.box_head]
            self.box_predictor = [box.box_predictor]
        else:
            self.box_head = box.box_head
            self.box_predictor = box.box_predictor
        self.box = box

    def forward(self, images, features, proposals, detections, targets=None, track_memory=None):
        losses = {}
        if self.training:
            _, tracks, loss_track = self.track(features, proposals, targets)
            losses.update(loss_track)
            return None, tracks, losses

        # solver is only needed during inference
        detections[0].set("state", torch.tensor([0 for _ in range(len(detections[0]))]).to(self.device))
        detections[0].set("ids", torch.tensor([-1 for _ in range(len(detections[0]))]).to(self.device))
        if not self.training:
            features_ = tuple([features[f] for f in features])
            y, tracks, _ = self.track(features_, detections, targets, track_memory)
            if tracks is not None:
                tracks = self._refine_tracks(images, features, tracks)
                detections = [Instances.cat(detections + tracks)]

            detections = self.solver(detections)
            x = self.track.get_track_memory(features_, detections)
            return x, detections, None

    def reset_roi_status(self):
        """
        Reset the status of ROI Heads
        """
        if self.cfg.MODEL.TRACK_ON:
            self.track.reset_track_pool()


    def _refine_tracks(self, images, features, tracks):
        """
        Use box head to refine the bounding box location
        The final vis score is an average between appearance and matching score
        """
        if len(tracks[0]) == 0:
            return tracks[0]

        track_scores = tracks[0].get("scores") + 1.
        tracks[0].set("proposal_boxes", tracks[0].pred_boxes)
        tracks[0].set("objectness_logits", tracks[0].scores)

        features = [features[f] for f in self.box.box_in_features]
        box_features = self.box.box_pooler(features, [x.pred_boxes for x in tracks])

        box_features = self.box_head[0](box_features)
        predictions = self.box_predictor[0](box_features)

        boxes = self.box_predictor[1].predict_boxes(predictions, tracks)
        scores = self.box_predictor[1].predict_probs(predictions, tracks)
        image_shapes = [x.image_size for x in tracks]

        pred_instances = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            0.5,
            0.5,
            100,
            tracks,
        )

        det_scores = pred_instances[0][0].get('scores')
        det_boxes = pred_instances[0][0].get('pred_boxes')

        if self.cfg.MODEL.TRACK_HEAD.TRACKTOR:
            scores = det_scores
        else:
            scores = (det_scores + track_scores) / 2
        # scores = track_scores
        r_tracks = Instances(tracks[0].image_size)
        r_tracks.set('pred_boxes', det_boxes) #tracks[0].pred_boxes) #tracks[0].pred_boxes)
        r_tracks.set('pred_classes', tracks[0].get('pred_classes'))
        r_tracks.set('scores', scores)
        r_tracks.set('ids', tracks[0].get('ids'))
        r_tracks.set("state", torch.tensor([0 for _ in range(len(r_tracks))]).to(self.device))

        return [r_tracks]

def build_track_heads(cfg, box):
    # individually create the heads, that will be combined together
    track_head = []

    if cfg.MODEL.TRACK_ON:
        track_utils, track_pool = build_track_utils(cfg)
        track_head.append(("track", build_track_head(cfg, track_utils, track_pool)))
        track_head.append(("solver", builder_tracker_solver(cfg, track_pool)))

    # combine individual heads in a single module
    if track_head:
        roi_heads = CombinedTrackHeads(cfg, track_head, box)
    else:
        roi_heads = None
    return roi_heads
