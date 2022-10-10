# -- coding: utf-8 --
import sympy as sp
k = sp.var('k',positive=True, integer=True)
a = sp.Matrix([[0, 1], [1, 1]])
val = a.eigenvals() #求特征值
vec = a.eigenvects() #求特征向量
P, D = a.diagonalize() #把a相似对角化
ak = P @ (D ** k) @ (P.inv())
F = ak @ sp.Matrix([1, 1])
s = sp.simplify(F[0])
print(s); sm = []
for i in range(20):
    sm.append(s.subs(k, i).n())
print(sm)