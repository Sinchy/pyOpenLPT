import numpy as np
from collections import defaultdict

def find_repeat_map(obj3d_list, eps=1e-3):
    # 提取三维坐标
    X = np.array([(o._pt_center[0], o._pt_center[1], o._pt_center[2])
                  for o in obj3d_list], dtype=np.float64)
    n = len(X)
    if n == 0:
        return {}

    # 分桶：按 eps 网格取 floor
    q = np.floor(X / eps).astype(np.int64)
    buckets = defaultdict(list)
    for i, key in enumerate(map(tuple, q)):
        buckets[key].append(i)

    # 邻域偏移（含自身桶，3x3x3）
    neigh = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    eps2 = eps*eps

    repeat_map = defaultdict(set)  # idx -> {重复它的其它 idx}
    for i in range(n):
        cx, cy, cz = q[i]
        xi = X[i]
        for dx,dy,dz in neigh:
            key = (cx+dx, cy+dy, cz+dz)
            for j in buckets.get(key, ()):
                if j <= i:  # 只检查 j>i，避免重复与自比
                    continue
                if np.dot(X[j]-xi, X[j]-xi) <= eps2:
                    repeat_map[i].add(j)
                    repeat_map[j].add(i)

    # 转为 list 并按索引排序
    return {i: sorted(list(js)) for i, js in sorted(repeat_map.items())}