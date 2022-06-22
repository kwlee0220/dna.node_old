from omegaconf import OmegaConf

from dna import Box, gdown_file
from .siammot_tracker import SiamMOT


_MODEL_URI = 'https://drive.google.com/u/0/uc?id=14Ci_qZFpN3i8HlwcO7PECCFbfVTQ7fJY'
def load(domain:Box, tracker_conf: OmegaConf):
    conf = tracker_conf.cnu_siammot
    
    model = conf.model
    gdown_file(_MODEL_URI, model)

    tracker = SiamMOT(model, conf.detectron2_conf, "cuda:0")
    return tracker
