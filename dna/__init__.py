from .color import BGR, WHITE, RED
from .types import Box, Size2d, Point, Image, Frame
from .utils import gdown_file, initialize_logger
from .pika_execution import PikaConnectionParameters, PikaExecutionServer
from .conf import *

__version__ = '0.0.7'

# DEBUG_FRAME_IDX = -1
# DEBUG_SHOW_IMAGE = False
# DEBUG_PRINT_COST = DEBUG_SHOW_IMAGE
# DEBUG_START_FRAME = 32
# DEBUG_TARGET_TRACKS = None
