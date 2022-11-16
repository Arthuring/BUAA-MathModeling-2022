# -- coding: utf-8 --
import random

import numpy
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

N = 100
h = 0.001
K = 9


def g(x1, a, b):
    return 1 / (b * np.sqrt(2 * np.pi)) * np.exp(-(x1 - a) ** 2 / (2 * b ** 2))


def y(x1):
    return g(x1, 0, 1) + g(x1, 1.5, 1)


def RMSE(predict_v, exact_v):
    sum = 0
    for i in range(len(exact_v)):
        sum += (predict_v[i] - exact_v[i]) ** 2
    return np.sqrt(sum / len(exact_v)).round(4)


def model21(t, a0, a1):
    return a0 + a1 * np.power(t, 1)

def model23(t, a0, a1, a2, a3):
    return a0 + a1 * np.power(t, 1) + a2 * np.power(t, 2) + + a3 * np.power(t, 3)

def model29(t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    return a0 + a1 * np.power(t, 1) + a2 * np.power(t, 2) + + a3 * np.power(t, 3) \
           + a4 * np.power(t, 4) + a5 * np.power(t, 5) + a6 * np.power(t, 6) \
           + a7 * np.power(t, 7) + a8 * np.power(t, 8) + a9 * np.power(t, 9)


X = numpy.linspace(0, 4, N).round(4)
Y = y(X)
Yr = Y + numpy.random.normal(0, np.sqrt(h), N)

index = [a for a in range(0, N)]
random.shuffle(index)
train = [[], [], []]
verify = [[], [], []]
test = [[], [], []]
for i in range(0, int(N * 0.8)):
    train[0].append(X[index[i]])
    train[1].append(Yr[index[i]])
    train[2].append(Y[index[i]])
for i in range(int(N * 0.8), int(N * 0.9)):
    verify[0].append(X[index[i]])
    verify[1].append(Yr[index[i]])
    verify[2].append(Y[index[i]])
for i in range(int(N * 0.9), int(N)):
    test[0].append(X[index[i]])
    test[1].append(Yr[index[i]])
    test[2].append(Y[index[i]])

p0 = np.random.randn(10)
print("参数初始值为：", p0)
p1 = curve_fit(model29, train[0], train[1], p0)[0]
print("参数拟合值为:", p1)
train_Yn = model29(train[0], *p1)
test_Yn = model29(test[0], *p1)
# -------------画图--------------------------
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), dpi=80)

Y_p = model29(X, *p1)

axes[0].set_title("Train")
axes[0].plot(X, Y_p, color='r', label='predict')
axes[0].scatter(train[0], train[1], label='exact')
axes[0].plot(X,Y,color='g', label='func')
train_r = RMSE(train[1], train_Yn)

print("训练集RMSE=", train_r)
axes[1].set_title("Test")
axes[1].plot(X, Y_p, color='r', label='predict')
axes[1].scatter(test[0], test[1], label='exact')
axes[1].plot(X,Y,color='g', label='func')
test_r = RMSE(test[1], test_Yn)
print("测试集RMSE=", test_r)

# figure.set_title("N="+str(N)+" h="+str(h))
axes[0].legend()  # 默认loc=Best
axes[1].legend()
axes[0].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' K=' + str(K) + ' RMSE=' + str(train_r) + '\nparameter final= ' + str(
        p1.round(4)))
axes[1].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' K=' + str(K) + ' RMSE=' + str(test_r) + '\nparameter finial= ' + str(
        p1.round(4)))

plt.show()
