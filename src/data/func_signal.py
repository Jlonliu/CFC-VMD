# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jialong Liu(jialong_liu@outlook.com)

import numpy as np

def sample_func(func, x_min, x_max, n=1000):
    x = np.linspace(x_min, x_max, n)
    try:
        y = np.asarray(func(x), dtype=float)   # 向量化路径（快）
    except Exception:
        y = np.array([func(xi) for xi in x])   # 逐点回退（慢但通用）
    return x, y

