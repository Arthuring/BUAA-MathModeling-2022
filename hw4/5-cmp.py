# -- coding: utf-8 --
import random

import numpy
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

N = 100
h = 0.001


def g(x1, a, b):
    return 1 / (b * np.sqrt(2 * np.pi)) * np.exp(-(np.add(x1, -a)) ** 2 / (2 * b ** 2))


def y(x1):
    return g(x1, 0, 1) + g(x1, 1.5, 1)


def RMSE(predict_v, exact_v):
    sum = 0
    for i in range(len(exact_v)):
        sum += (predict_v[i] - exact_v[i]) ** 2
    return np.sqrt(sum / len(exact_v)).round(4)


def model4(t):
    xx = np.multiply(-0.5824, t)
    return 0.3733 * np.cos(xx) + 0.2441


def model3(t):
    return -2.90721005e+00 * np.power(t, 0) * g(t, 1.36713804e+00, -1.44028058e+00) + \
           6.51344613e-01 * np.power(t, 1) * g(t, 1.36713804e+00, -1.44028058e+00) + \
           -3.14073751e+02 * np.power(t, 0) * g(t, -1.21483365e+01, 1.38207755e+00) + \
           -1.46424351e+04 * np.power(t, 1) * g(t, -1.21483365e+01, 1.38207755e+00)


def model2(t):
    return 0.49782277 + 0.34518524 * np.power(t, 1) + -0.27614233 * np.power(t, 2) + 0.04031836 * np.power(t, 3)


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

model2_Yn = model2(verify[0])
model3_Yn = model3(verify[0])
model4_Yn = model4(verify[0])
# -------------画图--------------------------
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), dpi=80)

Y_p2 = model2(X)
Y_p3 = model3(X)
Y_p4 = model4(X)

axes[0].set_title("model2")
axes[0].plot(X, Y_p2, color='r', label='predict')
axes[0].scatter(verify[0], verify[1], label='exact')
axes[0].plot(X, Y, color='g', label='func')
model2_r = RMSE(verify[2], model2_Yn)
print("model2 verify RMSE=", model2_r)

axes[1].set_title("model3")
axes[1].plot(X, Y_p3, color='r', label='predict')
axes[1].scatter(verify[0], verify[1], label='exact')
axes[1].plot(X, Y, color='g', label='func')
model3_r = RMSE(verify[2], model3_Yn)
print("model3 verify RMSE=", model3_r)

axes[2].set_title("Verify")
axes[2].plot(X, Y_p4, color='r', label='predict')
axes[2].scatter(verify[0], verify[1], label='exact')
axes[2].plot(X, Y, color='g', label='func')
model4_r = RMSE(verify[2], model4_Yn)
print("model4 verify RMSE=", model4_r)

# figure.set_title("N="+str(N)+" h="+str(h))
axes[0].legend()  # 默认loc=Best
axes[1].legend()
axes[0].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(
        model2_r) )
axes[1].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(
        model3_r) )
axes[2].set_xlabel(
    ' N=' + str(N) + ' h=' + str(h) + ' RMSE=' + str(
        model4_r) )

plt.show()

model2_Ynt = model2(test[0])
model3_Ynt = model3(test[0])
model4_Ynt = model4(test[0])

model2_rt = RMSE(test[2], model2_Ynt)
print("model2 test RMSE=", model2_rt)
model3_rt = RMSE(test[2], model3_Ynt)
print("model3 test RMSE=", model3_rt)
model4_rt = RMSE(test[2], model4_Ynt)
print("model4 test RMSE=", model4_rt)