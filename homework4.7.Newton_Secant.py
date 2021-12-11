# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 20:03:02 2021

@author: fengy
"""

import numpy as np
import sympy as sp

def Newton(funcs,args,x,tol = 1e-8):
    #funcs:待求根的方程组
    #args:符号变量
    #x:初始点
    #tol:停机条件
    err = 1
    n = x.shape[0]
    number = 0 #记录迭代次数
    Jcb = funcs.jacobian(args)
    while err>tol:
        subs_list = [(args[i],x[i,0]) for i in range(n)]
        J = np.array(Jcb.subs(subs_list),dtype=float)
        f = np.array(funcs.subs(subs_list),dtype=float)
        s = np.linalg.solve(J,-f).reshape(n,1)
        x = x+s
        err = np.linalg.norm(s)
        number = number+1
    return x,number

def Broyden(funcs,args,x,tol = 1e-8):
    #funcs:待求根的方程组
    #args:符号变量
    #x:初始点
    #tol:停机条件
    err = 1 
    number = 0 #记录迭代次数
    n = x.shape[0]
    B = np.eye(n) #B0取单位阵
    subs_list = [(args[i],x[i,0]) for i in range(n)]
    f = np.array(funcs.subs(subs_list),dtype=float) #初始化f
    while err>tol:
        s = np.linalg.solve(B,-f).reshape(n,1) #求解B_k*s_k = -f(x_k)
        x = x+s #更新x
        subs_list = [(args[i],x[i,0]) for i in range(n)]
        f_next = np.array(funcs.subs(subs_list),dtype=float)#求出f(x_{k+1})
        y = f_next-f 
        B = B + np.dot((y-np.dot(B,s)),s.reshape(1,n))/np.sum(s*s) #更新B
        f = f_next
        err = np.linalg.norm(s) 
        number = number+1
    return x,number

x1,x2 = sp.symbols('x1 x2')
args = sp.Matrix([x1,x2])
# f = sp.Matrix([x1 + 2*x2 - 2 , x1**2 + 4*x2**2 - 4])
# x = np.array([[1],[2]], dtype=float)
f = sp.Matrix([(x1 + 3)*(x2**3 - 7)+18, sp.sin(x2*sp.exp(x1)-1)])
x = np.array([[-0.5],[1.4]])
res1,t1 = Newton(f,args,x)
print('Newton:\n',np.around(res1, decimals=3))
print('Number of iterations:',t1)
res2,t2 = Broyden(f,args,x)
print('Broyden:\n',np.around(res2, decimals=3))
print('Number of iterations:',t2)

