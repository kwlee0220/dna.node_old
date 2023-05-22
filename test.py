from dna import Size2d


sz = Size2d([10, 20])
print(sz)

w, h = sz
print(w, h)

print(*sz)
print(tuple(sz))