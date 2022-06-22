import sys
sys.path.insert(0, './')

import argparse

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

from detectron2.data.datasets import register_coco_instances
from dna.track.centernet.config import add_centernet_config
from dna.track.siammot.config import add_siammot_config

parser = argparse.ArgumentParser(description="PyTorch SiamMOT Training")
parser.add_argument("--config-file", required=True, help="Training에서 사용할 detectron2 config 파일 위치", type=str)
parser.add_argument("--train-dir", required=True, help="Training 결과를 저장할 폴더 위치", type=str)
parser.add_argument("--device", default="cuda:0", help="학습 시 사용할 gpu 번호", type=str)

def setup_config(config_file):
    # build config file
    cfg = get_cfg()
    add_siammot_config(cfg)
    add_centernet_config(cfg)

    cfg.merge_from_file(config_file)

    cfg.DATASETS.TRAIN = ('coco_vehicle', 'bdd100k_train',)
    cfg.DATASETS.TEST = ('bdd100k_val',)

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.MAX_ITER = 500000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.SOLVER.BASE_LR = 0.002

    cfg.MODEL.TRACK_ON = False

    return cfg

if __name__ == '__main__':
    register_coco_instances('bdd100k_train', {}, "data/bdd100k/labels_coco/bdd100k_labels_train_coco_vehicle.json",
                            "data/bdd100k/data/images/100k/train")

    register_coco_instances('coco_vehicle', {}, "data/coco/annotations/coco_vehicle_only.json",
                            "data/coco/Images/train2017")

    register_coco_instances('mot17_train', {}, "data/MOT17_all/annotations/mot17_vehicle.json",
                            "data/MOT17_all/train")

    register_coco_instances('bdd100k_val', {}, "data/bdd100k/labels_coco/bdd100k_labels_val_coco_vehicle.json",
                            "data/bdd100k/data/images/100k/val")

    args = parser.parse_args()

    cfg = setup_config(args.config_file)
    cfg.OUTPUT_DIR = args.train_dir
    cfg.MODEL.DEVICE = args.device

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

