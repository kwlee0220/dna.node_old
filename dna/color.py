from collections import namedtuple

BGR = namedtuple('BGR', 'blue green red')
BLACK = BGR(0,0,0)
WHITE = BGR(255,255,255)
YELLOW = BGR(0,255,255)
RED = BGR(0, 0, 255)
PURPLE = BGR(128,0,128)
MAGENTA = BGR(255,0,255)
GREEN = BGR(0,255,0)
BLUE = BGR(255,0,0)
LIGHT_GREY = BGR(211, 211, 211)

import sys
def name_to_color(name: str):
    current_module = sys.modules[__name__]
    return current_module.__dict__[name]