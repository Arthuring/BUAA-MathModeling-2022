# -- coding: utf-8 --
import numpy as np
import pandas as pd
import pylab as plt

L = np.mat([[0.6, 0.1, 0.3],
           [0.1, 0.9, 0],
           [0.3, 0, 0.7]])

x_0 = np.mat([[2], [1], [1]])

val,vec = np.linalg.eig(L)
print("特征值：{}".format(val))
print("特征向量：{}".format(vec))

x_1 = []
x_2 = []
x_3 = []
x_k = []
limt = []
temp = x_0

for i in range(20):
    temp = L.dot(temp)
    x_k.append(temp.tolist())

    print("k = {}".format(i+1))
    print("x_k = {}".format(temp))
index = [i for i in range(20)]
df = pd.DataFrame(data={'x_k': x_k})
df.to_csv('./data.csv', sep=';')
temp = x_0
for i in range (100):
    temp = L.dot(temp)
    x_1.append(temp.tolist()[0][0])
    x_2.append(temp.tolist()[1][0])
    x_3.append(temp.tolist()[2][0])
    limt.append(4.0/3.0)

plt.xlabel("k")
plt.ylabel("y")
plt.title(r'Numerical solution of each component of $x_k$')
plt.plot(x_1,label=r'$x_k[1]$')
plt.plot(x_2,label=r'$x_k[2]$')
plt.plot(x_3,label=r'$x_k[3]$')
plt.plot(limt,'k--')
plt.legend(loc=1)
plt.text(60,1.35,r'$y = 4/3$')
plt.show()

