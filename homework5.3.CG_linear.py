# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:49:20 2021

@author: fengy
"""
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=8,suppress=True)


def CGLinear(A,b,x0,tol = 1e-8):
    '''
    共轭梯度法（Conjugate Gradient iethod）求解Ax=b
    
    Parameters
    ----------
    A : 系数矩阵
    b : 右端向量
    x : 初始点
    tol : 精度
    
    Returns
    -------
    x : Ax=b的解
    k : 迭代次数
    Res : 迭代中产生的残量
    '''
    x = x0
    r = b - A.dot(x)
    rk = np.sum(r*r)
    p = r 
    Res = r #记录残差向量
    k = 0 #记录迭代步数
    while rk**0.5>tol and k<1000:
        Ap = A.dot(p)
        alpha = rk/np.sum(p*Ap)
        x = x+alpha*p
        r = r-alpha*Ap
        Res = np.hstack([Res,r])
        rk1 = np.sum(r*r)
        beta = rk1/rk
        p = r+beta*p
        rk = rk1
        k = k+1
    return x,k,Res

if __name__ == '__main__': 
    
    #验证残量的正交性====================
    n = 10
    b = np.random.rand(n,1)
    x0 = np.zeros((n,1))
    A = np.random.rand(n,n) #生成随机矩阵
    A = np.dot(A.T,A)+4*np.eye(n) #生成对称正定矩阵
    x,k,Res = CGLinear(A,b,x0,1e-3)
    print(np.dot(Res.T,Res))
    
    #验证收敛速度和矩阵条件数的关系=====================
    n = 100
    rate = [0]*10
    number = [0]*10
    b = np.random.rand(n,1)
    x0 = np.zeros((n,1))
    for i in range(10):
        A = np.random.rand(n,n) #生成元素属于[0,1)的随机矩阵
        A = np.dot(A.T,A)+(i+1)*np.eye(n) #生成对称矩阵，加上(i+1)*单位阵减少病态性
        x,k,Res = CGLinear(A,b,x0) #共轭梯度法求解
        rate[i] = np.linalg.cond(A)**0.5
        number[i] = k
    #画出条件数的平方根与迭代步数的图像
    plt.plot(rate,number)    
    plt.show
    plt.xlabel('cond(A)^0.5')
    plt.ylabel('number of iterations')
    
    
