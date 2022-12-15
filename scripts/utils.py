
from omegaconf import OmegaConf

from dna.conf import get_config


def load_camera_conf(conf: OmegaConf, args_conf: OmegaConf) -> OmegaConf:
    conf.uri = get_config(args_conf, "camera", conf.get("uri"))
    conf.begin_frame = get_config(args_conf, "begin_frame", conf.get("begin_frame"))
    conf.end_frame = get_config(args_conf, "end_frame", conf.get("end_frame"))
    conf.sync = get_config(args_conf, "sync", conf.get("sync"))

    return conf