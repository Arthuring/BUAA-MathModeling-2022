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

model1 = lambda t, w1, a1, b1, w2, a2, b2: w1 * (
        1 / (b1 * np.sqrt(2 * np.pi)) * np.exp(-(t - a1) ** 2 / (2 * b1 ** 2))) + \
                                           w2 * (1 / (b2 * np.sqrt(2 * np.pi)) * np.exp(-(t - a2) ** 2 / (2 * b2 ** 2)))

p0 = np.random.randn(6).round(4)
print("参数初始值为：", p0)
p1 = curve_fit(model1, train[0], train[1], p0)[0]
print("参数拟合值为:", p1)
train_Yn = model1(train[0], *p1)
test_Yn = model1(test[0], *p1)
# -------------画图--------------------------
figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), dpi=80)

Y_p = model1(X, *p1)

axes[0][0].set_title("Train")
axes[0][0].plot(X, Y_p, color='r', label='predict')
axes[0][0].scatter(train[0], train[1], label='exact')
train_r = RMSE(train[1], train_Yn)

print("训练集RMSE=", train_r)
axes[0][1].set_title("Test")
axes[0][1].plot(X, Y_p, color='r', label='predict')
axes[0][1].scatter(test[0], test[1], label='exact')
test_r = RMSE(test[1], test_Yn)
print("测试集RMSE=", test_r)

# figure.set_title("N="+str(N)+" h="+str(h))
axes[0][0].legend()  # 默认loc=Best
axes[0][1].legend()
axes[0][0].set_xlabel(' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(train_r)+ '\nparameter initial= ' + str(p0))
axes[0][1].set_xlabel(' N=' + str(N) + ' h=' + str(h) + 'RMSE=' + str(test_r)+ '\nparameter initial= ' + str(p0))

p00 = [1, 1, 1, 1, 1, 1]
print("参数初始值为：", p00)
p11 = curve_fit(model1, train[0], train[1], p00)[0]
print("参数拟合值为:", p11)
train_Yn1 = model1(train[0], *p11)
test_Yn1 = model1(test[0], *p11)
# -------------画图--------------------------


Y_p = model1(X, *p11)

axes[1][0].set_title("Train")
axes[1][0].plot(X, Y_p, color='r', label='predict')
axes[1][0].scatter(train[0], train[1], label='exact')
train_r1 = RMSE(train[1], train_Yn1)

print("训练集RMSE=", train_r1)
axes[1][1].set_title("Test")
axes[1][1].plot(X, Y_p, color='r', label='predict')
axes[1][1].scatter(test[0], test[1], label='exact')
test_r1 = RMSE(test[1], test_Yn1)
print("测试集RMSE=", test_r1)

# figure.set_title("N="+str(N)+" h="+str(h))
axes[1][0].legend()  # 默认loc=Best
axes[1][1].legend()
axes[1][0].set_xlabel(' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(train_r1) + '\nparameter initial= ' + str(p00))
axes[1][1].set_xlabel(' N=' + str(N) + ' h=' + str(h) + 'RMSE=' + str(test_r1) + '\nparameter initial= ' + str(p00))

plt.show()
