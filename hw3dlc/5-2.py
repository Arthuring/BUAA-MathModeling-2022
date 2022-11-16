# -- coding: utf-8 --
import numpy as np
import cvxpy as cp
import gurobipy as gp
gp.setParam("TimeLimit", 600)
gp.setParam("MIPFocus", 3)


S = 2
K = [10, 15]
w = 0.5

N = 162

# 读取文件中的点
points = np.genfromtxt("points_xyz_" + str(N) + ".txt", dtype=float)

# 计算点之间的覆盖半径
d = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cos = np.round(points[i].dot(points[j]) / (np.linalg.norm(points[i]) * np.linalg.norm(points[j])), 4)
        d[i, j] = np.arccos(cos)

# h[i][j] = 1 表示第j号点选入第i组   h[i][j] = 0表示第j号点不选入第i组
h = cp.Variable((S, N), boolean=True)
# theta[0, S-1]为各组的最小覆盖半径，theta[S]是所有点的最小覆盖半径
theta = cp.Variable(S + 1, pos=True)

object = cp.Maximize(w / S * sum(theta[:-1]) + (1 - w) * theta[S])
conds = []
# theta 范围
for i in range(S + 1):
    conds.append(theta[i] >= 0)
    conds.append(theta[i] <= np.pi)
# 每组选择的点的数量
for i in range(S):
    conds.append(cp.sum(h[i]) == K[i])
# 一个点只能在一个组
for i in range(N):
    conds.append(cp.sum(h[:, i]) <= 1)

# 每组的theta大于等于组中各点间距
for i in range(S):
    for j in range(N):
        for k in range(j+1,N):
            conds.append(theta[i] - (1 - h[i, j]) * np.pi - (1 - h[i, k]) * np.pi <= d[j, k])

# 总体的theta大于等于各点间距
for i in range(N):
    for j in range(i + 1, N):
        conds.append(theta[S] - (1 - cp.sum(h[:, i])) * np.pi - (1 - cp.sum(h[:,j])) * np.pi <= d[i, j])

prob = cp.Problem(object, conds)
prob.solve(solver='GUROBI')

h_value = h.value
prob_value = prob.value

for i in range(S):
    print("\n第", i+1, "组选择：")
    for j in range(N):
        if h_value[i][j] == 1:
            print(j+1, " ", end="")

print("总最优值:", prob.value)