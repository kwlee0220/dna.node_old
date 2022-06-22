import sys
sys.path.insert(0, './')

import logging
import argparse
import os

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

from dna.track.siammot.config import add_siammot_config
from dna.track.siammot.engine.custom_trainer import CustomTrainer
from dna.track.centernet.config import add_centernet_config

parser = argparse.ArgumentParser(description="PyTorch SiamMOT Training")
parser.add_argument("--config-file", required=True, help="Training에서 사용할 detectron2 config 파일 위치", type=str)
parser.add_argument("--train-dir", required=True, help="Training 결과를 저장할 폴더 위치", type=str)
parser.add_argument("--models", default=None, help="학습 시 불러올 학습된 모델 파일", type=str)
parser.add_argument("--device", default="cuda:0", help="학습 시 사용할 gpu 번호", type=str)

def setup_config(config_file):
    cfg = get_cfg()
    add_siammot_config(cfg)
    add_centernet_config(cfg)

    cfg.merge_from_file(config_file)

    cfg.DATASETS.TRAIN = ("MOT_vehicle", "coco_vehicle", )
    cfg.DATASETS.TEST = ('bdd100k_val',)

    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = 8

    return cfg


def main():
    logger = logging.getLogger("detectron2")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        setup_logger()

    register_coco_instances('bdd100k_train', {}, "data/bdd100k/labels_coco/bdd100k_labels_train_coco_vehicle.json",
                            "data/bdd100k/data/images/100k/train")

    args = parser.parse_args()

    cfg = setup_config(args.config_file)
    cfg.OUTPUT_DIR = args.train_dir
    cfg.MODEL.WEIGHTS = args.models
    cfg.MODEL.DEVICE = args.device

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == "__main__":
    main()