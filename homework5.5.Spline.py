# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:28:15 2021

@author: fengy
"""
import numpy as np
import matplotlib.pyplot as plt


#=============打靶法===================
def PiecewisePoly(t,p,c,x):
    '''
    返回分段多项式在x处的函数值
    
    Parameters:
        t : 节点t0<t1<...<tn
        p : [t0,t1]上的三次多项式系数
        c : (t-ti)**3的系数
        
    Return:
        f : 函数值
    '''
    s = p[0]*x**3+p[1]*x**2+p[2]*x+p[3]
    n = len(c)
    for i in range(n):
        if x>t[i]:
            s = s+c[i]*(x-t[i])**3
    return s

def Shooting(t,y):
    '''
    打靶法求三次自然样条
    
    Parameters：
        t : 节点t0<t1<...<tn
        y : 节点处的函数值
        
    Returns:
        p : [t0,t1]上的三次多项式系数
        c : (t-ti)**3的系数，i=1,...,n-1
    '''
    n = len(t)
    A = np.array([[t[0]**3,   t[0]**2, t[0], 1],
                  [t[1]**3,   t[1]**2, t[1], 1],
                  [3*t[0]**2, 2*t[0],  1,    0],
                  [6*t[0],    2,       0,    0]])
    #假定s'(t[0])=1
    b = np.array([y[0],y[1],1,0])
    p1 = np.linalg.solve(A, b)
    c1 = np.zeros(n-1)
    for i in range(1,n-1):
        c1[i] = (y[i+1]-PiecewisePoly(t,p1,c1,t[i+1]))/(t[i+1]-t[i])**3
    #假定s'(t[0])=0    
    b = np.array([y[0],y[1],0,0])
    p2 = np.linalg.solve(A, b)
    c2 = np.zeros(n-1)
    for i in range(1,n-1):
        c2[i] = (y[i+1]-PiecewisePoly(t,p2,c2,t[i+1]))/(t[i+1]-t[i])**3
    #求三次样条在右端点处的二阶导数值
    DDs1 = 6*p1[0]*t[-1]+2*p1[1]+6*np.sum(c1*(t[-1]-t[:-1]))
    DDs2 = 6*p2[0]*t[-1]+2*p2[1]+6*np.sum(c2*(t[-1]-t[:-1]))
    if DDs1 == 0:
        p = p1
        c = c1
    elif DDs2 == 0:
        p = p2
        c = c2
    else:
        u = DDs2/(DDs2-DDs1)
        p = u*p1+(1-u)*p2
        c = u*c1+(1-u)*c2  
    return p,c

#=============B样条方法===================
def Bspline(i,k,t,x):
    '''
    求以t为节点，第i个k次B样条在x处的值
    '''
    if k==0:
        if t[i]<=x and x<t[i+1]:
            return 1
        else:
            return 0
   #如果有重节点，约定0/0=0
    if (t[i+k]-t[i])==0:
        alpha = 0
    else:
        alpha = (x-t[i])/(t[i+k]-t[i])
    if t[i+k+1]-t[i+1]==0:
         beta = 0
    else:
        beta = (t[i+k+1]-x)/(t[i+k+1]-t[i+1])
    #递推式   
    result = alpha*Bspline(i,k-1,t,x)+beta*Bspline(i+1,k-1,t,x)
    return result

def PlotBspline(L,R,n,k):
    '''
    画出区间[L，R]内等距节点的k次B样条基函数图像
 
    Parameters
    L,R : 区间端点
    n : 等分的小区间的个数
    k : B样条次数
    '''
    plt.figure()
    t = np.linspace(L,R,n+1) 
    h = (L-R)/n
    m = 10*n+1
    figt = np.linspace(t[0],t[-1],m) 
    #扩充节点
    h = (R-L)/n
    tt = np.hstack([t[0]-h*np.arange(k,0,-1),t,t[-1]+h*np.arange(1,k+1)])
    figy = np.zeros(m)
    for i in range(n+k):  
        for j in range(m):
            figy[j] = Bspline(i,k,tt,figt[j])
        plt.plot(figt,figy)
    plt.ylim(0,1)
    plt.legend(range(-k,n))
    s = 'B-splines of degree '+str(k)
    plt.title(s)
    plt.show()
    
def BsplineSolve(t,y,k):
    '''
    求出过数据点的样条函数在B样条基下的线性组合系数
    
    Parameters
    t : 插值节点
    y : 插值节点处的函数值
    k : B样条次数

    Returns
    c : 线性组合系数
    '''
    n = len(t)
    #扩充节点
    tt = np.hstack([t[0]-h*np.arange(k,0,-1),t,t[-1]+h*np.arange(1,k+1)])
    c = np.zeros(n+k-1)
    A = np.zeros((n+k-1,n+k-1))
    #过数据点
    for i in range(n):
        for j in range(i,i+3):
            A[i,j] = Bspline(j,k,tt,t[i]) 
    #端点处二阶导数为零      
    A[-2,[0,1,2]] = [1/h**2,-2/h**2,1/h**2]
    A[-1,[-3,-2,-1]] = [1/h**2,-2/h**2,1/h**2]
    yy = np.hstack([y,[0,0]])
    #求出系数
    c = np.linalg.solve(A,yy)
    return c
    
if __name__ == '__main__': 
    
    #测试打靶法=========
    t = np.array([1,3,5,9,10,15,18,20])
    y = np.random.randint(0,9,8)
    p,c = Shooting(t, y) #求分段多项式的系数
    figt = np.linspace(0,21,100)
    ShootingY = np.zeros_like(figt)   
    for i in range(100):
        ShootingY[i] = PiecewisePoly(t, p, c, figt[i]) 
    plt.figure()
    plt.scatter(t, y)
    plt.plot(figt,ShootingY)   
        
    L = 0 #区间左端点
    R = 2*np.pi #区间右端点
    n = 10 #区间内节点个数
        
    #画出1-4次B样条基函数图像================
    for k in range(1,5):
        PlotBspline(L,R,n,k)
    
    #随机生成节点处的函数值y,用打靶法和B样条方法求三次自然样条========
    m = 100*n
    t = np.linspace(L,R,n+1) 
    h = (R-L)/n
    y = np.random.randint(0,9,n+1)
    figt = np.linspace(L-h,R+h,m)
    #（1）B样条插值
    k = 3 #B样条次数
    c = BsplineSolve(t,y,k) #求B样条基函数的线性组合系数
    BsplineY = np.zeros_like(figt)
    tt = np.hstack([t[0]-h*np.arange(k,0,-1),t,t[-1]+h*np.arange(1,k+1)])
    for j in range(m):
        for i in range(n+k):  
            BsplineY[j] = BsplineY[j]+c[i]*Bspline(i,k,tt,figt[j]) 
    #（2）打靶法
    p,c = Shooting(t, y) #求分段多项式的系数
    ShootingY = np.zeros_like(figt)   
    for i in range(m):
        ShootingY[i] = PiecewisePoly(t, p, c, figt[i]) 
    # 画出两种方法生成的样条函数
    plt.figure()
    plt.scatter(t, y,c = 'y')
    plt.plot(figt,ShootingY,figt,BsplineY)
    plt.legend(['data points','Shooting','Bspline'])
    plt.show
   