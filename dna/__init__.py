from .color import BGR, WHITE, RED
from .types import Box, Size2d, Point, Image, Frame
from .utils import initialize_logger
from . import conf
from .pika_execution import PikaConnectionParameters, PikaExecutionServer

__version__ = '1.1.0'

DEBUG_FRAME_INDEX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = False
# DEBUG_START_FRAME = 32
# DEBUG_TARGET_TRACKS = None
