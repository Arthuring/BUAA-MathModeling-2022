
# -- coding: utf-8 --
import sympy as sp
import numpy as np
# 定义变量n
n = sp.var('n', positive=True, integer=True)
# 定义转移矩阵L
L = sp.Matrix([[sp.Rational(1,2) , 1],
               [1, 0]])
# 初始条件
x_0 = sp.Matrix([[0], [-2]])
# 求出L的特征值
val = L.eigenvals()
print("特征值：{}".format(val))
# 求出L的特征向量
vec = L.eigenvects()
print("特征向量：{}".format(vec))
# 相似对角化L
P, D = L.diagonalize()
print("相似对角化后P:")
print(P)
print("相似对角化后D:")
print(D)
# 计算L的n次方
Lk = P @ (D ** n) @ (P.inv())
# 带入初始值，求出最终解
F = Lk @ sp.Matrix([[0], [-2]])
print("最终解为：\n{}".format(F))

