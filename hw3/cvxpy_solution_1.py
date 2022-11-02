# -- coding: utf-8 --
import cvxpy as cp

x = cp.Variable((5), pos=True)

obj = cp.Minimize(20 * x[0] + 90 * x[1] + 80 * x[2] + 70 * x[3] + 30 * x[4])

cons = [
    x[0] + x[1] + x[4] >= 30.5,
    x[2] + x[3] >= 30,
    3 * x[0] + 2 * x[2] <= 120,
    3 * x[1] + 2 * x[3] + x[4] <= 48,
]

prob = cp.Problem(obj, cons)
prob.solve()
print("最优值为： ", prob.value)
print("最优解为： ", x.value)
