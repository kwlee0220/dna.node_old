import torch
import time

from detectron2.engine import DefaultTrainer
from ..data.build_train_data_loader import build_train_data_loader
import argparse
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.logger import setup_logger

from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase

class CustomTrainer(DefaultTrainer):
    def __int__(self, cfg):
        super().__int__()
        # logger = setup_logger()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        logger.info("Build Custom Trainer")
        model = build_model(cfg)
        model.to(cfg.MODEL.DEVICE)

        optimizer = build_optimizer(cfg, model)
        self.start_iter = 0

        data_loader = self.build_train_loader(cfg)
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        self.start_iter = 0

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        print("Build train loader")
        return build_train_data_loader(
            cfg,
            start_iter=0,
            shuffle=True
        )