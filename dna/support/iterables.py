
from typing import Union, List, Any, Tuple, TypeVar, Iterable, Generator, Optional, Callable, Dict

import itertools
from heapq import heappush
from collections import defaultdict

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

def get0(iterable): return iterable[0]
def get1(iterable): return iterable[1]

def mean(iterable) -> float:
    if not isinstance(iterable, list) and not isinstance(iterable, tuple):
        iterable = list(iterable)
    return sum(iterable) / len(iterable)

def first(iterable:Iterable[T], *, default:Optional[T]=None) -> Optional[T]:
    return next(iter(iterable), default)

def find(iterable:Iterable[T], key:K, *, keyer:Optional[Callable[[T],K]]=None) -> Optional[T]:
    for elm in iterable:
        item = keyer(elm) if keyer else elm
        if key == item:
            return elm
    return None

def find_cond(iterable:Iterable[T], cond:Callable[[T], bool]) -> Optional[T]:
    for elm in iterable:
        if cond(elm):
            return elm
    return None

def argfind(iterable:Iterable[T], key:K, *, keyer:Optional[Callable[[T],K]]=None) -> Tuple[int,Optional[T]]:
    for idx, elm in enumerate(iterable):
        item = keyer(elm) if keyer else elm
        if key == item:
            return idx, elm
    return -1, None

def flatmap(func, iterable):
    return itertools.chain.from_iterable(map(func, iterable))

def flatten(iterable):
    return itertools.chain.from_iterable(iterable)

def buffer(list:List[Any], count:int, skip:int=None, min_length:int=0) -> Generator[List[T], None, None]:
    if skip and skip <= 0:
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

def buffer_iterable(list:Iterable[T], count:int, skip:int=None) -> Generator[List[T], None, None]:
    skip = count if skip is None else skip
    if skip <= 0:
        raise ValueError(f"invalid skip: {skip}")
    
    iterator = iter(list)
    buffer = []
    try:
        while True:
            buffer.append(next(iterator))
            if len(buffer) >= count:
                yield buffer.copy()
                buffer = buffer[skip:]
    except StopIteration as expected:
        if buffer:
            yield buffer
            
def groupby(list:Iterable[T], key:Callable[[T],K], *, value:Callable[[T],V]=lambda t: t) -> Dict[K,List[Union[T,V]]]:
    groups:Dict[K,List[T]] = dict()
    for v in list:
        grp_key = key(v)
        if grp_key not in groups:
            groups[grp_key] = []
        groups[grp_key].append(value(v))
    return groups

def difference(left:Iterable[T], right:Iterable[T], *, key:Callable[[T],V]=lambda t:t) -> Generator[T, None, None]:
    return (v1 for v1 in left if find(right, key=key(v1), keyer=key) is None)

if __name__ == '__main__':
    list = list(range(10))
    for b in buffer(list, 3, 2, min_length=2):
        print(b)