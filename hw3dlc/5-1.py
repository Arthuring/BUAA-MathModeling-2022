# -- coding: utf-8 --
import math

import numpy as np
import cvxpy as cp
import gurobipy as gp

gp.setParam("TimeLimit", 600)
gp.setParam("MIPFocus", 1)


K = 30
N = 162

# 读取文件中的点
points = np.genfromtxt("points_xyz_" + str(N) + ".txt", dtype=float)

# 计算点之间的覆盖半径
d = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cos = np.round(points[i].dot(points[j]) / (np.linalg.norm(points[i]) * np.linalg.norm(points[j])), 4)
        d[i, j] = np.arccos(cos)

# 定义变量 h可取0，1， 取1表示选择该点， 取0表示不选该点
h = cp.Variable(N, boolean=True)
theta = cp.Variable(1, pos=True)

# 优化目标
object = cp.Maximize(theta[0])

# 限制条件 总点数为k，theta范围[0,pi]
conds = [sum(h) == K,
         theta[0] >= 0,
         theta[0] <= math.pi]
# h可取0，1
# for i in range(N):
#     conds.append(h[i] >= 0)
#     conds.append(h[i] <= 1)

# theta >= d[i, j]
for i in range(N):
    for j in range(i + 1, N):
        conds.append(theta[0] - (1 - h[i]) * math.pi - (1 - h[j]) * math.pi <= d[i, j])

# 求解
prob = cp.Problem(object, conds)
prob.solve(solver='GUROBI', verbose=True, MIPFocus=1)

# 输出
ans = np.nonzero(h.value)
theta_max = theta[0].value

print("选择点：", ans)
print("最优值：", theta_max)
