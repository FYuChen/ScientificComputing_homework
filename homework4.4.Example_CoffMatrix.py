# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:48:25 2021

@author: fengy
"""
import numpy as np
from matplotlib import pyplot as plt
#差分法求解二阶微分方程的边值问题
#(D(x)u'(x))'=b(x),x in [a,b]

def D1(x):
    return 1
def D2(x):
    return np.exp(x)

def coff(D,N,a,b):
    #a,b 是区间的左右端点
    #N+1是节点数
    #D是公式中的D(x)
    A = np.zeros((N+1,N+1))
    x = np.linspace(a,b,N+1)
    for i in range (1,N):
        A[i,i-1] = D((x[i]+x[i-1])/2)
        A[i,i] = -D((x[i]+x[i-1])/2)-D((x[i+1]+x[i])/2)
        A[i,i+1] = D((x[i+1]+x[i])/2)
    A = A[1:N,1:N]
    return A

A1 = coff(D1,100,-1,1)
w1, v1 = np.linalg.eig(A1) 
idx = w1.argsort()[::-1]  #特征值从大到小排序 
w1 = w1[idx]
v1 = v1[:,idx]

A2 = coff(D2,100,-5,5)
w2, v2 = np.linalg.eig(A2)
idx = w2.argsort()[::-1]  #特征值从大到小排序 
w2 = w2[idx]
v2 = v2[:,idx]

x = np.arange(1,100,1).reshape(99,1)
plt.figure(1,figsize=(8,4))
plt.subplot(1,  2,  1) 
plt.scatter(x,w1,s=10)
plt.title('D(x)=1')
plt.subplot(1,  2,  2) 
plt.scatter(x,w2,s=10)
plt.title('D(x)=exp(x)')
print('cond(A1)=',w1[0]/w1[-1])
print('cond(A2)=',w2[0]/w2[-1])
plt.show()

A3 = np.zeros_like(A2)
for i in range(99):
    A3[[i],:] = A2[[i],:]/A2[i,i]
w3, v3 = np.linalg.eig(A3)
idx = w3.argsort()[::-1]  #特征值从大到小排序 
w3 = w3[idx]
v3 = v3[:,idx]
plt.figure(2)
plt.scatter(x,w3,s=10)
plt.title('D(x)=exp(x),divided')
plt.show()
print('cond(A3)=',w3[0]/w3[-1])



    




