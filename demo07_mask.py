# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_mask.py  掩码
"""
import numpy as np

# 基于bool数组的掩码
a = np.arange(1, 100)
# 输出100以内3的倍数
mask = (a % 3 == 0) & (a % 7 == 0)
print(mask)
print(a[mask])

# 基于索引的掩码
names = np.array(['Apple', 'Mate30 pro',
                  'MI', 'Oppo', 'Vivo'])
rank = [1, 0, 3, 4, 2]
print(names[rank])
