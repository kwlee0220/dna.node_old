from detectron2.config import CfgNode as CN

def add_siammot_config(cfg):
    _C = cfg

    _C.MODEL.RPN.USE_FPN = True
    _C.MODEL.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
    _C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
    _C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 2000
    _C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
    _C.MODEL.RPN.POST_NMS_TOP_N_TEST = 300
    _C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 300

    _C.MODEL.ROI_HEADS.USE_FPN = True
    _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    _C.MODEL.BOX_ON = True
    _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    _C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
    _C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
    _C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FPNPredictor"
    _C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
    _C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024

    # DLA
    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.DLA_STAGE2_OUT_CHANNELS = 64
    _C.MODEL.DLA.DLA_STAGE3_OUT_CHANNELS = 128
    _C.MODEL.DLA.DLA_STAGE4_OUT_CHANNELS = 256
    _C.MODEL.DLA.DLA_STAGE5_OUT_CHANNELS = 512
    _C.MODEL.DLA.BACKBONE_OUT_CHANNELS = 128
    _C.MODEL.DLA.STAGE_WITH_DCN = (False, False, False, False, False, False)

    # TRACK branch
    _C.MODEL.TRACK_ON = True
    _C.MODEL.EMBED_ON = True
    _C.MODEL.TRACK_HEAD = CN()
    _C.MODEL.TRACK_HEAD.TRACKTOR = False
    _C.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
    _C.MODEL.TRACK_HEAD.POOLER_RESOLUTION = 15
    _C.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO = 2

    _C.MODEL.TRACK_HEAD.PAD_PIXELS = 512
    # the times of width/height of search region comparing to original bounding boxes
    _C.MODEL.TRACK_HEAD.SEARCH_REGION = 2.0
    # the minimal width / height of the search region
    _C.MODEL.TRACK_HEAD.MINIMUM_SREACH_REGION = 0
    _C.MODEL.TRACK_HEAD.MODEL = 'EMM'

    # solver params
    _C.MODEL.TRACK_HEAD.TRACK_THRESH = 0.4
    _C.MODEL.TRACK_HEAD.START_TRACK_THRESH = 0.6
    _C.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = 0.4
    # maximum number of frames that a track can be dormant
    _C.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES = 1

    # track proposal sampling
    _C.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE = 256
    _C.MODEL.TRACK_HEAD.PREDICTOR_CHANNELS = 256
    _C.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD = 0.65
    _C.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD = 0.35

    _C.MODEL.TRACK_HEAD.IMM = CN()
    # the feature dimension of search region (after fc layer)
    # in comparison to that of target region (after fc layer)
    _C.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM_MULTIPLIER = 2
    _C.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM = 256

    _C.MODEL.TRACK_HEAD.EMM = CN()
    # Use_centerness flag only activates during inference
    _C.MODEL.TRACK_HEAD.EMM.USE_CENTERNESS = True
    _C.MODEL.TRACK_HEAD.EMM.POS_RATIO = 0.25
    _C.MODEL.TRACK_HEAD.EMM.HN_RATIO = 0.25
    _C.MODEL.TRACK_HEAD.EMM.TRACK_LOSS_WEIGHT = 1.
    # The ratio of center region to be positive positions
    _C.MODEL.TRACK_HEAD.EMM.CLS_POS_REGION = 0.8
    # The lower this weight, it allows large motion offset during inference
    # Setting this param to be small (e.g. 0.1) for datasets that have fast motion,
    # such as caltech roadside pedestrian
    _C.MODEL.TRACK_HEAD.EMM.COSINE_WINDOW_WEIGHT = 0.4
    _C.MODEL.DETECTRON2_FORMAT = False

    # all video-related parameters
    _C.VIDEO = CN()
    # the length of video clip for training/testing
    _C.VIDEO.TEMPORAL_WINDOW = 8
    # the temporal sampling frequency for training
    _C.VIDEO.TEMPORAL_SAMPLING = 4
    _C.VIDEO.RANDOM_FRAMES_PER_CLIP = 2

    # Inference
    _C.INFERENCE = CN()
    _C.INFERENCE.USE_GIVEN_DETECTIONS = False
    # The length of clip per forward pass
    _C.INFERENCE.CLIP_LEN = 1

    # Solver
    _C.SOLVER.CHECKPOINT_PERIOD = 5000
    _C.SOLVER.VIDEO_CLIPS_PER_BATCH = 16

    # Input
    cfg.INPUT.MOTION_LIMIT = 0.1
    cfg.INPUT.COMPRESSION_LIMIT = 50
    cfg.INPUT.MOTION_BLUR_PROB = 0.5
    cfg.INPUT.AMODAL = False
    cfg.INPUT.BRIGHTNESS = 0.1
    cfg.INPUT.CONTRAST = 0.1
    cfg.INPUT.SATURATION = 0.1
    cfg.INPUT.HUE = 0.1

    # Root directory of datasets
    _C.DATASETS.ROOT_DIR = ''
    _C.DATALOADER.SIZE_DIVISIBILITY = 32

    # Distance inference
    _C.EMBED_DISTANCE = 'center'
