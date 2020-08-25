# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
how_to_use.py  Numpyの基本の使い方
"""

import numpy as np

#NumPy配列を生成する
np.array([1, 2, 3])
# array([1, 2, 3])

# 多次元配列を作成
np.array([[1, 2, 3], [4, 5, 6]])
# array([[1, 2, 3],
#       [4, 5, 6]])


# 配列について確認する

# 配列の要素型を調べる(dtype)
# **Numpyのデータ型**

# |  型名         |  型表示符                           |
# | ------------  | ----------------------------------- |
# | ブーリン      | bool_                               |
# | 符号整数      | int8(-128~127)/int16/int32/int64    |
# | 符号無し整数  | uint8(0~255)/uint16/uint32/uint64   |
# | フロート      | float16/float32/float64             |
# | 複素数        | complex64/complex128                |
# | 文字列        | str_，每个字符用32位Unicode编码表示  |

x = np.array([1, 2, 3])
x.dtype
# dtype('int64')

# 配列の構造を調べる(shape)
x = np.array([1, 2, 3])
x.shape
# (3,)

# 全体の要素数を調べる(size)
x = np.array([1, 2, 3])
x.size
# 3

# 配列に要素を追加する
x = np.array([1, 2, 3])
print(x)
# [1 2 3]

y = np.append(x, [4, 5, 6])
print(y)
# [1 2 3 4 5 6]

# 配列の次元を変換する
x = np.array([1, 2, 3, 4, 5, 6])
print(x)
# [1 2 3 4 5 6]

y = x.reshape(2, -1)
print(y)
# [[1 2 3]
#  [4 5 6]]


# 配列の要素にアクセスする
# NumPy配列には、Pythonのリストと同じように、インデックス番号でアクセスできます。
# 書き方：[開始:終了:間隔]
x = np.array(range(1, 10))
print(x)
# [1 2 3 4 5 6 7 8 9]

print(x[1::2])
# [2 4 6 8]


a = np.arange(1, 10)
print(a)  # [1 2 3 4 5 6 7 8 9]
print(a[:3])  # 1 2 3
print(a[3:6])   # 4 5 6
print(a[6:])  # 7 8 9
print(a[::-1])  # 9 8 7 6 5 4 3 2 1
print(a[:-4:-1])  # 9 8 7
print(a[-4:-7:-1])  # 6 5 4
print(a[-7::-1])  # 3 2 1
print(a[::])  # 1 2 3 4 5 6 7 8 9
print(a[:])  # 1 2 3 4 5 6 7 8 9
print(a[::3])  # 1 4 7
print(a[1::3])  # 2 5 8
print(a[2::3])  # 3 6 9

# マスク機能
a = np.arange(1, 10)
mask = [True, False,True, False,True, False,True, False,True]
print(a[mask])
# [1 3 5 7 9]

# 間隔にマイナスを指定すると最後の要素から順に取り出されます。
x = np.array(range(1, 10))
print(x)
# [1 2 3 4 5 6 7 8 9]

print(x[::-2])
# [9 7 5 3 1]

# Pythonのリストと同じように、条件式で抽出することもできます。
x = np.array(range(1, 10))
print(x)
# [1 2 3 4 5 6 7 8 9]

print(x[x > 5])
# [6 7 8 9]


#要素をループする
a = np.array([[[1, 2],
               [3, 4]],
              [[5, 6],
               [7, 8]]])
print(a, a.shape)
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [7 8]]] (2, 2, 2)

print(a[0])
# [[1 2]
#  [3 4]]

print(a[0][0])
# [1 2]

print(a[0][0][0])
# 1

print(a[0, 0, 0])
# 1

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            print(a[i, j, k])
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8



# 等間隔な数列を生成する
# 等間隔な数列を生成するには、linspace命令を使用します。
# 書き方：np.linspace(開始, 終了, num=数列数)
np.linspace(0, 10, num=101)
# array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
#         1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
#         2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
#         3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
#         4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
#         5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
#         6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
#         7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
#         8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
#         9.9, 10. ])

# 乱数を生成する
# 乱数を生成するには、random.rand命令、またはrandom.randn命令を使用します。
# randは一様乱数を、randnは標準正規分布乱数を生成します
# 書き方：random.rand(要素数, 次元数)

np.random.rand(10, 1)
# array([[0.74952345],
#        [0.36226121],
#        [0.38173576],
#        [0.14002005],
#        [0.03540485],
#        [0.05020347],
#        [0.71791232],
#        [0.45254545],
#        [0.5176943 ],
#        [0.44750293]])

# 配列の要素を混ぜる
x = np.array(range(1, 10))
np.random.shuffle(x)
print(x)
# [6 3 8 7 2 1 4 5 9]

# 配列を並び替える
# 書き方：sort(配列)
# ・昇順にソート
x = np.array(range(1, 10))
np.random.shuffle(x)
print(x)
# [6 3 8 7 2 1 4 5 9]

y = np.sort(x)
print(y)
# [1 2 3 4 5 6 7 8 9]

# ・降順にソート
x = np.array(range(1, 10))
np.random.shuffle(x)
print(x)
# [4 3 7 1 9 8 6 2 5]

y = np.sort(x)[::-1]
print(y)
# [9 8 7 6 5 4 3 2 1]


# 配列の要素型を変える
ary = np.array([1, 2, 3, 4, 5, 6])
print(type(ary), ary, ary.dtype)
# <class 'numpy.ndarray'> [1 2 3 4 5 6] int64

#int→float
b = ary.astype(float)　
print(type(b), b, b.dtype)
# <class 'numpy.ndarray'> [1. 2. 3. 4. 5. 6.] float64

#int→string
c = ary.astype(str)
print(type(c), c, c.dtype)
# <class 'numpy.ndarray'> ['1' '2' '3' '4' '5' '6'] <U21



# 配列の要素型を宣言する
data=[
	('zs', [90, 80, 85], 15),
	('ls', [92, 81, 83], 16),
	('ww', [95, 85, 95], 15)
]

# １番目の方法
# U3　Unicode３つ
# 3int32　int32が３つ

a = np.array(data, dtype='U3, 3int32, int32')
print(a)
# [('zs', [90, 80, 85], 15) ('ls', [92, 81, 83], 16)
#  ('ww', [95, 85, 95], 15)]

print(a[0]['f0'], ":", a[1]['f1'])
# zs : [92 81 83]
print("=====================================")


# ２番目の方法
b = np.array(data, dtype=[('name', 'str_', 2),
                    ('scores', 'int32', 3),
                    ('ages', 'int32', 1)])
print(b[0]['name'], ":", b[0]['scores'])
# zs : [90 80 85]

print("=====================================")

# ３番目の方法
c = np.array(data, dtype={'names': ['name', 'scores', 'ages'],
                    'formats': ['U3', '3int32', 'int32']})
print(c[0]['name'], ":", c[0]['scores'], ":", c.itemsize)
# zs : [90 80 85] : 28

print("=====================================")

# 日付
f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
y = f.astype('M8[Y]')
print(y)
# ['2011' '2012' '2013' '2011']

f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
d = f.astype('M8[D]')
print(d)
# ['2011-01-01' '2012-01-01' '2013-01-01' '2011-02-01']


f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
m = f.astype('M8[M]')
print(m)
# ['2011-01' '2012-01' '2013-01' '2011-02']


f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
h = f.astype('M8[h]')
print(h)
# ['2011-01-01T00' '2012-01-01T00' '2013-01-01T01' '2011-02-01T00']

f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
mi = f.astype('M8[m]')
print(mi)
# ['2011-01-01T00:00' '2012-01-01T00:00' '2013-01-01T01:01'
#  '2011-02-01T00:00']


f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
s = f.astype('M8[s]')
print(s)
# ['2011-01-01T00:00:00' '2012-01-01T00:00:00' '2013-01-01T01:01:01'
#  '2011-02-01T00:00:00']


f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
f = f.astype('M8[D]')
f = f.astype('int32')
print(f)
# [14975 15340 15706 15006]

print(f[3]-f[0])
# 31

print("=====================================")


# 結合と分割
# 縦方向
a = np.arange(1, 7).reshape(2, 3)
print(a)
# [[1 2 3]
#  [4 5 6]]

b = np.arange(7, 13).reshape(2, 3)
print(b)
# [[ 7  8  9]
#  [10 11 12]]

# 縦方向結合
c = np.vstack((a, b))
print(c)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

# 縦方向分割
d, e = np.vsplit(c, 2)
print(d)
# [[1 2 3]
#  [4 5 6]]

print(e)
# [[ 7  8  9]
#  [10 11 12]]


# 横方向
a = np.arange(1, 7).reshape(2, 3)
print(a)
# [[1 2 3]
#  [4 5 6]]

b = np.arange(7, 13).reshape(2, 3)
print(b)
# [[ 7  8  9]
#  [10 11 12]]

# 横方向結合
c = np.hstack((a, b))
print(c)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]

# 横方向分割
d, e = np.hsplit(c, 2)
print(d)
# [[1 2 3]
#  [4 5 6]]

print(e)
# [[ 7  8  9]
#  [10 11 12]]

# 奥方向
a = np.arange(1, 7).reshape(2, 3)
print(a)
# [[1 2 3]
#  [4 5 6]]

b = np.arange(7, 13).reshape(2, 3)
print(b)
# [[ 7  8  9]
#  [10 11 12]]

# 奥方向結合
i = np.dstack((a, b))
print(i)
# [[[ 1  7]
#   [ 2  8]
#   [ 3  9]]

#  [[ 4 10]
#   [ 5 11]
#   [ 6 12]]]

# 奥方向分割
k, l = np.dsplit(i, 2)
print(k)
# [[[1]
#   [2]
#   [3]]

#  [[4]
#   [5]
#   [6]]]

print(l)
# [[[ 7]
#   [ 8]
#   [ 9]]

#  [[10]
#   [11]
#   [12]]]


# もっと簡単に書く結合と分割
# 結合
# 0：縦
# 1：横
# 2：奥

np.concatenate((a, b), axis=0)
np.split(c, 2, axis=0)


# 異なるサイズの結合
a = np.array([1,2,3,4,5])
b = np.array([1,2,3,4])
b = np.pad(b, pad_width=(0, 1), mode='constant', constant_values=-1)
print(b)
# [ 1  2  3  4 -1]

c = np.vstack((a, b))
print(c)
# [[ 1  2  3  4  5]
#  [ 1  2  3  4 -1]]

# 列と行の結合
a = np.arange(1,9)		#[1, 2, 3, 4, 5, 6, 7, 8]
b = np.arange(9,17)		#[9,10,11,12,13,14,15,16]

c = np.row_stack((a, b))
print(c)
# [[ 1  2  3  4  5  6  7  8]
#  [ 9 10 11 12 13 14 15 16]]


d = np.column_stack((a, b))
print(d)
# [[ 1  9]
#  [ 2 10]
#  [ 3 11]
#  [ 4 12]
#  [ 5 13]
#  [ 6 14]
#  [ 7 15]
#  [ 8 16]]


# Numpy配列属性
# - shape - 维度
# - dtype - 元素类型
# - size - 元素数量
# - ndim - 维数，len(shape)
# - itemsize - 元素字节数
# - nbytes - 总字节数 = size x itemsize
# - real - 复数数组的实部数组
# - imag - 复数数组的虚部数组
# - T - 数组对象的转置视图
# - flat - 扁平迭代器
a = np.array([[1 + 1j, 2 + 4j, 3 + 7j],
              [4 + 2j, 5 + 5j, 6 + 8j],
              [7 + 3j, 8 + 6j, 9 + 9j]])
print(a.shape)
# (3, 3)
print(a.dtype)
# complex128

print(a.ndim)
# 2
print(a.size)
# 9
print(a.itemsize)
# 16
print(a.nbytes)
# 144
print(a.real, a.imag, sep='\n')
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [7. 8. 9.]]
# [[1. 4. 7.]
#  [2. 5. 8.]
#  [3. 6. 9.]]

print(a.T)
# [[1.+1.j 4.+2.j 7.+3.j]
#  [2.+4.j 5.+5.j 8.+6.j]
#  [3.+7.j 6.+8.j 9.+9.j]]

print([elem for elem in a.flat])
# [(1+1j), (2+4j), (3+7j), (4+2j), (5+5j), (6+8j), (7+3j), (8+6j), (9+9j)]

b = a.tolist()
print(b)
# [[(1+1j), (2+4j), (3+7j)], [(4+2j), (5+5j), (6+8j)], [(7+3j), (8+6j), (9+9j)]]
