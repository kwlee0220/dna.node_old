from boltons import iterutils

x0 = range(10)
# print(list(x0))

# x1 = iterutils.chunked(x0, 3, count=2)
# print(x1)

# x2 = iterutils.chunked(x0, 3, fill=999)
# print(x2)

# x1 = iterutils.chunked_iter(x0, 3)
# print(list(x1))

# xx = list(range(10))
# x3 = [xx[slice(*range)] for range in iterutils.chunk_ranges(input_size=10, chunk_size=3, overlap_size=1)]
# print(x3)

# x = list(iterutils.chunk_ranges(input_offset=4, input_size=15, chunk_size=5, overlap_size=2, align=False))
# print(x)

pleasantries = ['hi', 'hello', 'ok', 'bye', 'yes']
print(iterutils.unique(pleasantries, key=lambda x: len(x)))