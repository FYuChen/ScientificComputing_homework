# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:49:20 2021

@author: fengy
"""
import numpy as np
import sympy as sp
np.set_printoptions(precision=3,suppress=True)

def PhiValue(f,args,x,s,alpha):
    '''
    求phi(alpha)=f(x+alpha*s)在alpha处的值
    '''
    n = x.shape[0]
    subs_list = [(args[i],x[i,0]+alpha*s[i,0]) for i in range(n)]
    phi = np.array(f.subs(subs_list),dtype=float)[0,0]
    return phi
   
def GoldenSection(f,args,x,s,R):
    '''
    线搜索求步长
    用黄金分割法求phi(alpha)=f(x+alpha*s)在[0,R]上的极小值点

    Parameters
    ----------
    f : 目标函数
    args : 符号变量
    x : 当前迭代点
    s : 方向
    R : 右端点
    
    Returns
    -------
    alpha : 最优步长
    '''
    b = R
    a = 0
    tau = (5**0.5-1)/2
    alpha1 = a+(1-tau)*(b-a)
    phi1 = PhiValue(f,args,x,s,alpha1)
    alpha2 = a+tau*(b-a)
    phi2 = PhiValue(f,args,x,s,alpha2)
    while (b-a)>1e-7:
        if phi1>phi2:
            a = alpha1
            alpha1 = alpha2
            phi1 = phi2
            alpha2 = a+tau*(b-a)
            phi2 = PhiValue(f, args, x, s, alpha2)
        else:
            b = alpha2
            alpha2 = alpha1
            phi2 = phi1
            alpha1 = a+(1-tau)*(b-a)
            phi1 = PhiValue(f, args, x, s, alpha1)
    return (a+b)/2
    
def CGLineSearch(f,args,x0):
    '''
    共轭梯度法求f的极小值

    Parameters
    ----------
    f : 目标函数
    args : 符号变量
    x0 : 初始点

    Returns
    -------
    x : 近似解
    '''
    df = sp.Matrix([f.diff(t) for t in args]) #f的梯度
    n = x0.shape[0]
    x = x0
    subs_list = [(args[i],x0[i,0]) for i in range(n)]
    g = np.array(df.subs(subs_list),dtype=float) #梯度在x0处的值
    s = -g #初始方向为负梯度方向
    gk = np.sum(g*g)
    alpha = 1
    k = 0
    print('k = 0: ')
    print('x =\n',x0)
    print('gradient f = ',df)
    print('g =\n',g)
    print('s =\n',s)
    while gk>1e-7 and k<1000:
        alpha = GoldenSection(f,args,x,s,2*alpha) #黄金分割法求一维极小化问题，求出步长
        x = x+alpha*s #更新x
        subs_list = [(args[i],x[i,0]) for i in range(n)]
        g = np.array(df.subs(subs_list),dtype=float) #更新梯度g
        gk1 = np.sum(g*g)
        beta = gk1/gk 
        gk = gk1
        s = -g+beta*s #更新方向s
        k = k+1
        print('\nk = ',k)
        print('x =\n',x)
        print('alpha = %.3f'%alpha)
        print('g =\n',g)
        print('beta = %.3f'%beta)
        print('s = \n',s)
    return x,k

if __name__ == '__main__':    
     
    #测试：课本P283.Example 6.14 =======
    x1,x2 = sp.symbols('x1 x2')
    args = sp.Matrix([x1,x2])
    f = sp.Matrix([0.5*x1**2 + 2.5*x2**2])
    x0 = np.array([[5],[1.0]])
    x,k = CGLineSearch(f,args,x0)
