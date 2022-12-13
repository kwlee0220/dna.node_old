
from typing import List, Any

def buffer(list:List[Any], count:int, skip:int=None, min_length:int=0):
    if skip <= 0:
        raise ValueError(f"invalid skip: {skip}")
    if min_length > count:
        raise ValueError(f"invalid: min_length({min_length}) > count({count})")
    
    idx = 0
    length = len(list)
    skip = count if skip is None else skip
    while idx < length:
        upper = min(length, idx + count)
        if upper - idx < min_length:
            break
        yield list[idx:upper]
        idx += skip

if __name__ == '__main__':
    list = list(range(10))
    for b in buffer(list, 3, 2, min_length=2):
        print(b)