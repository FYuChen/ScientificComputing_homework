# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:57:07 2021

@author: fengy
"""
import numpy as np
import matplotlib.pyplot as plt


def LagrangeInter(t,y,x):
    '''
    拉格朗日插值
    
    Parameters:
        t : 节点t0<t1<...<tn
        y : 节点处函数值
        x : 待求值的点
        
    Returns：
        s: 拉格朗日插值多项式在x处的值
    '''
    n = len(t)    
    s = 0                                   
    for i in range(n):
        Li=y[i] 
    #构造基函数
        for j in range(n):
           if j!=i:
            Li=Li*(x-t[j])/(t[i]-t[j])      
        s=s+Li                        
    return s


if __name__ == '__main__': 
    L = -1
    R = 1
    figt = np.linspace(L,R,100)
    figy = np.zeros_like(figt) 
    for n in [6,10,14]: 
        t = np.linspace(L,R,n) 
        y = 1/(1+25*t**2)
        for i in range(100):
            figy[i] = LagrangeInter(t,y,figt[i])
        s = 'n='+str(n) 
        plt.plot(figt,figy,label=s)
        plt.scatter(t, y)    
    yy = 1/(1+25*figt**2)
    plt.plot(figt,yy,label = 'f=1/(1+25*t^2)')
    plt.legend()
    plt.show()
