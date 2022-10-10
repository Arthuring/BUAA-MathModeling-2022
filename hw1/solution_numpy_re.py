# -- coding: utf-8 --
# -- coding: utf-8 --
import sympy as sp
import numpy as np

l = sp.var('l')
exp = (l - sp.Rational(6, 10)) * (l - sp.Rational(9, 10)) * (l - sp.Rational(7, 10)) - sp.Rational(9, 100) * (
        l - sp.Rational(9, 10)) - sp.Rational(1, 100) * (l - sp.Rational(7, 10))
expp = sp.simplify(exp)

print("特征方程为：{}".format(expp))
k = sp.var('k', positive=True, integer=True)

L = sp.Matrix([[0.6, 0.1, 0.3],
           [0.1, 0.9, 0],
           [0.3, 0, 0.7]])

Lp = np.mat([[0.6, 0.1, 0.3],
           [0.1, 0.9, 0],
           [0.3, 0, 0.7]])
x_0 = sp.Matrix([[2], [1], [1]])
val, vec = np.linalg.eig(Lp)
print("特征值：{}".format(val))

print("特征向量：{}".format(vec))

P, D = L.diagonalize()  # 相似对角化
print("相似对角化后P:")
print(P)
print("相似对角化后D:")
print(D)

Lk = P @ (D ** k) @ (P.inv())  # 计算L的k次方
F = Lk @ np.array([[2], [1], [1]])  # 求出最终解x_k
print("最终解为：\n{}".format(F))
# ---------计算稳定解-------------------
Pinv = P.inv()  # 计算P的逆矩阵
c = (Pinv @ x_0)[0]  # 计算c的值
print("c={}".format(c))
print("稳定解为：\n{}".format(c * sp.Matrix([[1],[1],[1]])))
