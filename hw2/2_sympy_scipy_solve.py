# -- coding: utf-8 --
import sympy as sp
from scipy.integrate import odeint
import numpy as np
import pylab as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# -----------求解析解-----------
# 定义自变量和函数
t = sp.var('t')
x = sp.Function('x')
y = sp.Function('y')
# 列出微分方程组
eq = (sp.Eq(sp.Derivative(x(t), t, 1), x(t) - 2 * y(t)), sp.Eq(sp.Derivative(y(t), t, 1), x(t) + 2 * y(t)))
# 列出初始条件
con = {y(0): 0, x(0): 1}
# 求解析解
result = sp.dsolve(eq, ics=con)
print(result)
sx = sp.lambdify(t, result[0].args[1], 'numpy')
sy = sp.lambdify(t, result[1].args[1], 'numpy')

# -----------求数值解-----------
dz = lambda z, t: [z[0] - 2 * z[1], z[0] + 2 * z[1]]
t = np.linspace(0, 1, 100)
s = odeint(dz, [1, 0], t)

# -----------画图-----------
# 画x-t图
plt.plot(t, sx(t), 'g-')
plt.plot(t, s[:, 0], '.')
plt.xlabel('$t$')
plt.ylabel('$x$', rotation=0)
plt.legend(['[0,1]上$x(t)$的解析解', '[0,1]上$x(t)$的数值解'])
plt.show()
# 画y-t图
plt.plot(t, sy(t), 'g-')
plt.plot(t, s[:, 1], '.')
plt.xlabel('$t$')
plt.ylabel('$y$', rotation=0)
plt.legend(['[0,1]上$y(t)$的解析解', '[0,1]上$y(t)$的数值解'])
plt.show()
# 画y-x图
plt.title('y(x)图')
plt.plot(s[:, 0], s[:, 1])
plt.xlabel('$x$')
plt.ylabel('$y$', rotation=0)
plt.show()
