import torch
import copy

from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, Instances, pairwise_iou

class EMMTargetSampler(object):
    """
    Sample track targets for SiamMOT.
    It samples from track proposals from RPN
    """

    def __init__(self, cfg, track_utils, matcher, propsals_per_image=256,
                 pos_ratio=0.25, hn_ratio=0.25):
        self.cfg = cfg
        self.track_utils = track_utils
        self.proposal_iou_matcher = matcher
        self.proposals_per_image = propsals_per_image
        self.hn_ratio = hn_ratio
        self.pos_ratio = pos_ratio

    def match_targets_with_iou(self, proposal: Instances, gt: Instances):
        match_quality_matrix = pairwise_iou(gt.bbox, proposal.bbox)
        matched_idxs = self.proposal_iou_matcher(match_quality_matrix)

        target = copy.copy(gt)  # .copy_with_fields(("ids", "labels"))
        matched_target = target[torch.clamp_min(matched_idxs[0], -1)]
        proposal_ids = matched_target.get("ids")
        proposal_labels = matched_target.get("labels")
        # proposal_gt_boxes = matched_target.get("gt_boxes").tensor

        # id = -1 for background
        # id = -2 for ignore proposals
        proposal_ids[matched_idxs[1] == -1] = -1
        proposal_ids[matched_idxs[1] == 0] = -2
        proposal_labels[matched_idxs[1] <= 0] = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES

        return proposal_ids.type(torch.int64), proposal_labels.type(torch.int64)

    def assign_matched_ids_to_proposals(self, proposals: Instances, gts: Instances):
        """
        Assign for each proposal a matched id, if it is matched to a gt box
        Otherwise, it is assigned -1
        """
        for proposal, gt in zip(proposals, gts):
            if len(gt) == 0:
                device = gt.bbox.device
                temp_gt = Instances(gt.image_size)
                temp_gt.set("bbox", Boxes(torch.tensor([[-1, -1, -1, -1]])).to(device))
                temp_gt.set("labels", torch.tensor([-1], dtype=torch.int64).to(device))
                temp_gt.set("ids", torch.tensor([-1], dtype=torch.int64).to(device))
                temp_gt.to(device)
                proposal_ids, proposal_labels = self.match_targets_with_iou(proposal, temp_gt)
            else:
                proposal_ids, proposal_labels = self.match_targets_with_iou(proposal, gt)

            proposal.set('ids', proposal_ids)
            proposal.set('labels', proposal_labels)

    def duplicate_boxlist(self, boxlist, num_duplicates):
        """
        Duplicate samples in box list by concatenating multiple times.
        """
        if num_duplicates == 0:
            return self.get_dummy_boxlist(boxlist)
        list_to_join = []
        for _ in range(num_duplicates):
            # dup = boxlist.copy_with_fields(list(boxlist.extra_fields.keys()))
            list_to_join.append(boxlist)

        return Instances.cat(list_to_join)

    def get_dummy_boxlist(self, boxlist: Instances, num_boxes=0):
        """
        Create dummy boxlist, with bbox [-1, -1, -1, -1],
        id -1, label -1
        when num_boxes = 0, it means return an empty BoxList
        """
        boxes = Boxes(torch.zeros((num_boxes, 4)) - 1.)
        dummy_boxlist = self.get_default_boxlist(boxlist, boxes)

        return dummy_boxlist

    @staticmethod
    def get_default_boxlist(boxlist: Instances, bboxes, ids=None, labels=None):
        """
        Construct a boxlist with bbox as bboxes,
        all other fields to be default
        id -1, label -1
        """
        device = boxlist.bbox.tensor.device
        num_boxes = bboxes.tensor.shape[0]
        if ids is None:
            ids = torch.zeros((num_boxes,), dtype=torch.int64) - 1
        if labels is None:
            labels = torch.zeros((num_boxes,), dtype=torch.int64) - 1

        default_boxlist = Instances(boxlist.image_size)
        default_boxlist.set('bbox', bboxes)
        default_boxlist.set('labels', labels)
        default_boxlist.set('ids', ids)

        return default_boxlist.to(device)

    @staticmethod
    def sample_examples(src_box: [Instances], pair_box: [Instances],
                        tar_box: [Instances], num_samples, gt=None):
        """
        Sample examples
        """
        if len(src_box) == 0:
            device = gt.bbox.device
            temp_instances = Instances(gt.image_size)
            temp_instances.set("bbox", Boxes(torch.tensor([])).to(device))
            temp_instances.set("labels", torch.tensor([], dtype=torch.int64).to(device))
            temp_instances.set("ids", torch.tensor([], dtype=torch.int64).to(device))
            return [temp_instances, temp_instances, temp_instances]

        src_box = Instances.cat(src_box)
        pair_box = Instances.cat(pair_box)
        tar_box = Instances.cat(tar_box)

        assert (len(src_box) == len(pair_box) and len(src_box) == len(tar_box))

        if len(src_box) <= num_samples:
            return [src_box, pair_box, tar_box]
        else:
            indices = torch.zeros((len(src_box),), dtype=torch.bool)
            permuted_idxs = torch.randperm(len(src_box))
            sampled_idxs = permuted_idxs[: num_samples]
            indices[sampled_idxs] = 1

            sampled_src_box = src_box[indices]
            sampled_pair_box = pair_box[indices]
            sampled_tar_box = tar_box[indices]
            return [sampled_src_box, sampled_pair_box, sampled_tar_box]

    def sample_boxlist(self, boxlist: Instances, indices, num_samples):
        assert (num_samples <= indices.numel())

        if num_samples == 0:
            sampled_boxlist = self.get_dummy_boxlist(boxlist, num_boxes=0)
        else:
            permuted_idxs = torch.randperm(indices.numel())
            sampled_idxs = indices[permuted_idxs[: num_samples], 0]
            sampled_boxes = boxlist.bbox[sampled_idxs, :]
            sampled_ids = None
            sampled_labels = None
            if boxlist.has("ids"):
                sampled_ids = boxlist.get('ids')[sampled_idxs]
            if boxlist.has("labels"):
                sampled_labels = boxlist.get('labels')[sampled_idxs]

            sampled_boxlist = self.get_default_boxlist(boxlist, sampled_boxes,
                                                       sampled_ids, sampled_labels)
        return sampled_boxlist

    def get_target_box(self, target_gt, indices):
        """
        Get the corresponding target box given by the 1-off indices
        if there is no matching target box, it returns a dummy box
        """
        tar_box = target_gt[indices]
        # the assertion makes sure that different boxes have different ids
        assert (len(tar_box) <= 1)
        if len(tar_box) == 0:
            # dummy bounding boxes
            tar_box = self.get_dummy_boxlist(target_gt, num_boxes=1)

        return tar_box

    def generate_hn_pair(self, src_gt: Instances, proposal: Instances,
                         src_h=None, proposal_h=None):
        """
        Generate hard negative pair by sampling non-negative proposals
        """
        proposal_ids = proposal.get('ids')
        src_id = src_gt.get('ids')

        scales = torch.ones_like(proposal_ids)
        if (src_h is not None) and (proposal_h is not None):
            scales = src_h / proposal_h

        # sample proposals with similar scales
        # and non-negative proposals
        hard_bb_idxs = ((proposal_ids >= 0) & (proposal_ids != src_id))
        scale_idxs = (scales >= 0.5) & (scales <= 2)
        indices = (hard_bb_idxs & scale_idxs)
        unique_ids = torch.unique(proposal_ids[indices])
        idxs = indices.nonzero()

        # avoid sampling redundant samples
        num_hn = min(idxs.numel(), unique_ids.numel())
        sampled_hn_boxes = self.sample_boxlist(proposal, idxs, num_hn)

        return sampled_hn_boxes

    def generate_pos(self, src_gt: Instances, proposal: Instances):
        assert (len(src_gt) == 1)
        proposal_ids = proposal.get('ids')
        src_id = src_gt.get('ids')

        pos_indices = (proposal_ids == src_id)
        pos_boxes = proposal[pos_indices]

        return pos_boxes

    def generate_pos_hn_example(self, proposals, gts):
        """
        Generate positive and hard negative training examples
        """
        src_gts = copy.deepcopy(gts)
        tar_gts = self.track_utils.swap_pairs(copy.deepcopy(gts))

        track_source = []
        track_target = []
        track_pair = []
        for src_gt, tar_gt, proposal in zip(src_gts, tar_gts, proposals):
            pos_src_boxlist, pos_pair_boxlist, pos_tar_boxlist = ([] for _ in range(3))
            hn_src_boxlist, hn_pair_boxlist, hn_tar_boxlist = ([] for _ in range(3))

            proposal_h = proposal.bbox.tensor[:, 3] - proposal.bbox.tensor[:, 1]

            src_gt_ = src_gt[src_gt.ids > -1]
            tar_gt_ = tar_gt[tar_gt.ids > -1]

            src_h = src_gt_.bbox.tensor[:, 3] - src_gt_.bbox.tensor[:, 1]
            src_ids = src_gt_.get('ids')
            tar_ids = tar_gt_.get('ids')

            for i, src_id in enumerate(src_ids):
                _src_box = src_gt_[src_ids == src_id]
                _tar_box = self.get_target_box(tar_gt_, tar_ids == src_id)

                pos_src_boxes = self.generate_pos(_src_box, proposal)
                pos_pair_boxes = copy.deepcopy(pos_src_boxes)
                pos_tar_boxes = self.duplicate_boxlist(_tar_box, len(pos_src_boxes))

                hn_pair_boxes = self.generate_hn_pair(_src_box, proposal, src_h[i], proposal_h)
                hn_src_boxes = self.duplicate_boxlist(_src_box, len(hn_pair_boxes))
                hn_tar_boxes = self.duplicate_boxlist(_tar_box, len(hn_pair_boxes))

                pos_src_boxlist.append(pos_src_boxes)
                pos_pair_boxlist.append(pos_pair_boxes)
                pos_tar_boxlist.append(pos_tar_boxes)

                hn_src_boxlist.append(hn_src_boxes)
                hn_pair_boxlist.append(hn_pair_boxes)
                hn_tar_boxlist.append(hn_tar_boxes)

            num_pos = int(self.proposals_per_image * self.pos_ratio)
            num_hn = int(self.proposals_per_image * self.hn_ratio)
            sampled_pos = self.sample_examples(pos_src_boxlist, pos_pair_boxlist,
                                               pos_tar_boxlist, num_pos, src_gt_)
            sampled_hn = self.sample_examples(hn_src_boxlist, hn_pair_boxlist,
                                              hn_tar_boxlist, num_hn, src_gt_)
            for pos in sampled_pos:
                if pos.has("objectness"):
                    pos.remove('objectness')

            track_source.append(Instances.cat([sampled_pos[0], sampled_hn[0]]))
            track_pair.append(Instances.cat([sampled_pos[1], sampled_hn[1]]))
            track_target.append(Instances.cat([sampled_pos[2], sampled_hn[2]]))

        return track_source, track_pair, track_target

    def generate_neg_examples(self, proposals: [Instances], gts: [Instances], pos_hn_boxes: [Instances]):
        """
        Generate negative training examples
        """
        track_source = []
        track_pair = []
        track_target = []
        for proposal, gt, pos_hn_box in zip(proposals, gts, pos_hn_boxes):
            proposal_ids = proposal.get('ids')
            objectness = proposal.get('objectness')

            proposal_h = proposal.bbox.tensor[:, 3] - proposal.bbox.tensor[:, 1]
            proposal_w = proposal.bbox.tensor[:, 2] - proposal.bbox.tensor[:, 0]

            neg_indices = ((proposal_ids == -1) & (objectness >= 4.0) &
                           (proposal_h >= 5) & (proposal_w >= 5))
            idxs = neg_indices.nonzero()

            neg_samples = min(idxs.numel(), self.proposals_per_image - len(pos_hn_box))
            neg_samples = max(0, neg_samples)

            sampled_neg_boxes = self.sample_boxlist(proposal, idxs, neg_samples)
            # for target box
            sampled_tar_boxes = self.get_dummy_boxlist(proposal, neg_samples)

            track_source.append(sampled_neg_boxes)
            track_pair.append(sampled_neg_boxes)
            track_target.append(sampled_tar_boxes)
        return track_source, track_pair, track_target

    def __call__(self, proposals: [Instances], gts: [Instances], images=None):
        sampler_proposals = []
        for proposal in proposals:
            temp_ = Instances(proposal.image_size)
            temp_.set("bbox", proposal.get('proposal_boxes'))
            temp_.set("objectness", proposal.get('objectness_logits'))
            sampler_proposals.append(temp_)

        sampler_gts = []
        for gt in gts:
            temp_ = Instances(gt.image_size)
            temp_.set("bbox", gt[gt.ids > 0].get("gt_boxes"))
            temp_.set("labels", gt[gt.ids > 0].get("gt_classes"))
            temp_.set("ids", gt[gt.ids > 0].get("ids"))
            sampler_gts.append(temp_)

        self.assign_matched_ids_to_proposals(sampler_proposals, sampler_gts)

        pos_hn_src, pos_hn_pair, pos_hn_tar = self.generate_pos_hn_example(sampler_proposals, sampler_gts)
        neg_src, neg_pair, neg_tar = self.generate_neg_examples(sampler_proposals, sampler_gts, pos_hn_src)

        track_source = [Instances.cat([pos_hn, neg]) for (pos_hn, neg) in zip(pos_hn_src, neg_src)]
        track_target = [Instances.cat([pos_hn, neg]) for (pos_hn, neg) in zip(pos_hn_tar, neg_tar)]

        sr = self.track_utils.update_boxes_in_pad_images(track_source)
        sr = self.track_utils.extend_bbox(sr)
        return track_source, sr, track_target


def make_emm_target_sampler(cfg,
                            track_utils):
    matcher = Matcher(
        [cfg.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD,
         cfg.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD],
        [-1, 0, 1],
        allow_low_quality_matches=False,
    )

    track_sampler = EMMTargetSampler(cfg, track_utils, matcher,
                                     propsals_per_image=cfg.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE,
                                     pos_ratio=cfg.MODEL.TRACK_HEAD.EMM.POS_RATIO,
                                     hn_ratio=cfg.MODEL.TRACK_HEAD.EMM.HN_RATIO,
                                     )
    return track_sampler