# -- coding: utf-8 --
import numpy as np
import cvxpy as cp
import gurobipy as gp

w = 0.5  # 权重
S = 2  # 组数
K = [10, 15]

N = 162 # 总的点数
M = np.pi  # 一个极大的常数

# 算出任意两个向量的arccos,并存入d保留四位小数
coordinate = np.genfromtxt("points_xyz_" + str(N) + ".txt", dtype=float)
d = np.ones((N, N)) * np.pi
for i in range(N):
    for j in range(i + 1, N):
        d[i][j] = np.arccos(np.round(coordinate[i] @ coordinate[j], 4))

h = cp.Variable((S, N), integer=True)  # 每组一个h[i],h[i].len = N
theta = cp.Variable(S + 1, pos=True)  # theta[0,S-1] 属于各组  theta[S]是全部组合并

obj = cp.Maximize(w / S * sum(theta[:-1]) + (1 - w) * theta[S])

cons = [h >= 0, h <= 1, theta >= 0, theta <= np.pi]

for i in range(S):
    cons.append(cp.sum(h[i]) == K[i])  # 每组的点数固定

for i in range(N):
    cons.append(cp.sum(h[:, i]) <= 1)  # 每个点最多出现在一个组中

for s in range(S):
    for i in range(N):
        for j in range(i + 1, N):
            cons.append(theta[s] - M * (1 - h[s, i]) - M * (1 - h[s, j])
                        <= d[i, j])

for i in range(N):
    for j in range(i + 1, N):
        cons.append(theta[S] - M * (1 - cp.sum(h[:, i])) - M * (1 - cp.sum(h[:, j]))
                    <= d[i, j])

prob = cp.Problem(obj, cons)
gp.setParam("TimeLimit", 600)
gp.setParam("MIPFocus", 3)
prob.solve(solver='GUROBI')
print("选择的点是:")
h_value = h.value
for i in range(S):
    for j in range(N):
        if h_value[i, j] > 0.9:
            print("第" + str(i + 1) + "组" + ",包括第" + str(j + 1) + "个点")
print(prob.value)
