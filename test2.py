import numpy as np
from dna import Size2d, Point


pt = Point([1.2, 4.7])
print(pt)
print(np.array(pt))
print(np.array(pt, int))

sz = Size2d([10.2, 20.7])
print(sz)

print(sz + 1)
sz2 = sz.to_rint()
print(sz2.wh.dtype)