# -- coding: utf-8 --
import random

import numpy
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

N = 100
h = 0.001


def g(x1, a, b):
    return 1 / (b * np.sqrt(2 * np.pi)) * np.exp(-(x1 - a) ** 2 / (2 * b ** 2))


def y(x1):
    return g(x1, 0, 1) + g(x1, 1.5, 1)


def RMSE(predict_v, exact_v):
    sum = 0
    for i in range(len(exact_v)):
        sum += (predict_v[i] - exact_v[i]) ** 2
    return np.sqrt(sum / len(exact_v)).round(4)


def model4(t, a, b, c):
    xx = np.multiply(b,t)
    return a * np.cos(xx) + c


def model4b(t, a, c):
    xx = np.multiply(10000, t)
    return a * np.cos(xx) + c


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

p0 = [1,1]
print("参数初始值为：", p0)
p1 = curve_fit(model4b, train[0], train[1], p0)[0]
print("参数拟合值为:", p1)
train_Yn = model4b(train[0], *p1)
test_Yn = model4b(test[0], *p1)
verify_Yn = model4b(verify[0], *p1)
# -------------画图--------------------------
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), dpi=80)

Y_p = model4b(X, *p1)

axes[0].set_title("Train")
axes[0].plot(X, Y_p, color='r', label='predict')
axes[0].scatter(train[0], train[1], label='exact')
axes[0].plot(X, Y, color='g', label='func')
train_r = RMSE(train[1], train_Yn)
print("训练集RMSE=", train_r)

axes[1].set_title("Test")
axes[1].plot(X, Y_p, color='r', label='predict')
axes[1].scatter(test[0], test[1], label='exact')
axes[1].plot(X, Y, color='g', label='func')
test_r = RMSE(test[1], test_Yn)
print("测试集RMSE=", test_r)

axes[2].set_title("Verify")
axes[2].plot(X, Y_p, color='r', label='predict')
axes[2].scatter(verify[0], verify[1], label='exact')
axes[2].plot(X, Y, color='g', label='func')
verify_r = RMSE(test[1], verify_Yn)
print("验证集RMSE=", verify_r)

# figure.set_title("N="+str(N)+" h="+str(h))
axes[0].legend()  # 默认loc=Best
axes[1].legend()
axes[0].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(
        train_r) + '\nparameter final= ' + str(
        p1.round(4)))
axes[1].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(
        test_r) + '\nparameter finial= ' + str(
        p1.round(4)))
axes[2].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(
        verify_r) + '\nparameter finial= ' + str(
        p1.round(4)))

plt.show()
