import torch

from detectron2.structures import Boxes, Instances


class TrackHead(torch.nn.Module):
    def __init__(self, cfg, tracker, tracker_sampler, track_utils, track_pool):
        super(TrackHead, self).__init__()
        self.device = cfg.MODEL.DEVICE
        self.tracker = tracker
        self.sampler = tracker_sampler

        self.track_utils = track_utils
        self.track_pool = track_pool

    def forward(self, features, proposals=None, targets=None, track_memory=None):
        if self.training:
            return self.forward_train(features, proposals, targets)
        else:
            return self.forward_inference(features, track_memory=track_memory)

    def forward_train(self, features, proposals=None, targets=None):
        """
        Perform correlation on feature maps and regress the location of the object in other frame
        :param features: a list of feature maps from different intermediary layers of feature backbone
        :param proposals:
        :param targets:
        """
        with torch.no_grad():
            track_proposals, sr, track_targets = self.sampler(proposals, targets)

        return self.tracker(features, track_proposals, sr=sr, targets=track_targets)

    def forward_inference(self, features, track_memory=None):
        track_boxes = None
        if track_memory is None:
            self.track_pool.reset()
        else:
            (template_features, sr, template_boxes) = track_memory
            del track_memory
            if template_features.numel() > 0:
                return self.tracker(features, [template_boxes[0].to(self.device)], sr=[sr[0].to(self.device)],
                                    template_features=template_features.to(self.device))
        return {}, track_boxes, {}


    def cuda_memory(self, track_memory):
        (template_features, sr, template_boxes) = track_memory
        return template_features.to(self.device), [sr[0].to(self.device)], [template_boxes[0].to(self.device)]

    def reset_roi_status(self):
        """
        Reset the status of ROI Heads
        """
        if self.cfg.MODEL.TRACK_ON:
            self.track.reset_track_pool()

    def get_track_memory(self, features, tracks):
        assert (len(tracks) == 1)
        active_tracks = self._get_track_targets(tracks[0])

        # no need for feature extraction of search region if
        # the tracker is tracktor, or no trackable instances
        if len(active_tracks) == 0:
            template_features = torch.tensor([], device=features[0].device)
            sr = Instances([active_tracks.image_size[0] + self.track_utils.pad_pixels * 2,
                            active_tracks.image_size[1] + self.track_utils.pad_pixels * 2])
            for field_ in active_tracks.get_fields():
                sr.set(field_, active_tracks.get(field_))
            track_memory = (template_features, [sr], [active_tracks])
        else:
            track_memory = self.tracker.extract_cache(features, active_tracks)

        track_memory = self._update_memory_with_dormant_track(track_memory)

        self.track_pool.update_cache(track_memory)

        return track_memory

    def _update_memory_with_dormant_track(self, track_memory):
        cache = self.track_pool.get_cache()
        if not cache or track_memory is None:
            return track_memory

        dormant_caches = []
        for dormant_id in self.track_pool.get_dormant_ids():
            if dormant_id in cache:
                dormant_caches.append(cache[dormant_id])
        cached_features = [x[0][None, ...] for x in dormant_caches]
        if track_memory[0] is None:
            if track_memory[1][0] or track_memory[2][0]:
                raise Exception("Unexpected cache state")
            track_memory = [[]] * 3
            buffer_feat = []
        else:
            buffer_feat = [track_memory[0]]
        features = torch.cat(buffer_feat + cached_features)
        sr = Instances.cat(track_memory[1] + [x[1] for x in dormant_caches])
        boxes = Instances.cat(track_memory[2] + [x[2] for x in dormant_caches])
        return features, [sr], [boxes]

    def _get_track_targets(self, target):
        if len(target) == 0:
            return target
        active_ids = self.track_pool.get_active_ids()

        ids = target.get('ids').tolist()
        idxs = torch.zeros((len(ids), ), dtype=torch.bool, device=target.pred_boxes.tensor.device)
        for _i, _id in enumerate(ids):
            if _id in active_ids:
                idxs[_i] = True

        return target[idxs]


def build_track_head(cfg, track_utils, track_pool):

    from ..track_head.EMM.track_core import EMM
    from ..track_head.EMM.target_sampler import make_emm_target_sampler

    tracker = EMM(cfg, track_utils)

    tracker_sampler = make_emm_target_sampler(cfg, track_utils)

    return TrackHead(cfg, tracker, tracker_sampler, track_utils, track_pool)
