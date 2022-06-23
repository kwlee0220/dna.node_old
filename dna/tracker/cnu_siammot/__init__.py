from omegaconf import OmegaConf
import pathlib

from dna import Box, gdown_file
from .siammot_tracker import SiamMOT


_MODEL_PATH = 'models/siam_mot.pth'
_MODEL_URI = 'https://drive.google.com/u/0/uc?id=14Ci_qZFpN3i8HlwcO7PECCFbfVTQ7fJY'
_DETECTOR_CONF_FILE = 'conf/CenterNet_siammot.yaml'

def load(domain:Box, tracker_conf: OmegaConf):
    conf = tracker_conf.get('cnu_siammot', OmegaConf.create())
    
    model_path = conf.get('model', _MODEL_PATH)
    gdown_file(_MODEL_URI, model_path)
    
    detector_conf = conf.get('detector', None)
    if detector_conf is None:
        plugin_dir = pathlib.Path(__file__).parent.resolve()
        detector_conf = plugin_dir / _DETECTOR_CONF_FILE

    tracker = SiamMOT(model_path, detector_conf, "cuda:0")
    return tracker
