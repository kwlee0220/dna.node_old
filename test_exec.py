
import threading
from contextlib import closing
import json
import logging

from datetime import timedelta
import cv2
import numpy as np
from omegaconf import OmegaConf

import dna
from dna import Box
from dna.camera import create_opencv_camera_from_conf
from dna.camera.image_processor import ImageProcessor
from dna.execution import LoggingExecutionContext
from dna.support.rectangle_drawer import RectangleDrawer

dna.utils.initialize_logger()
logger = logging.getLogger("dna.test")

json_obj = None
with open('data/on-demand-requests/req01_etri_05.json') as f:
    json_obj = json.load(f)
conf = OmegaConf.create(json_obj)
conf.parameters.id = conf.id
conf.parameters.progress_report = conf.progress_report
conf = conf.parameters
conf.show_progress = True
conf.window_name = 'test'

proc = None
def do_work(conf):
    global proc
    camera = create_opencv_camera_from_conf(conf.camera)
    proc = ImageProcessor(camera.open(), conf, LoggingExecutionContext(logger))
    proc.report_interval = 3
    proc.run()
    cv2.destroyAllWindows()    

t1 = threading.Thread(target=do_work, args=(conf,))
t1.start()
t1.join(timeout=10)
proc.stop()
print('done')