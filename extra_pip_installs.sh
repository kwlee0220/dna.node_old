#! /bin/bash

# PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# Detectron2 (Linux Only)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# CNU SiamMOT dependencies
pip install gluoncv mxnet imgaug