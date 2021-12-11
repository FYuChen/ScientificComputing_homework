
import numpy as np
np.set_printoptions(precision=3,suppress=True,threshold=3)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

def Arnoldi(A,N):
    '''
    Arnoldi迭代
    Parameters:
        A:任意矩阵
        N:Kroylov子空间维数
    Return:
        H:上Hessenberg矩阵
    '''
    n = A.shape[0]
    Q = np.zeros((n,N))
    H = np.zeros((N,N))
    x = np.ones((n,1))/n**0.5 
    Q[:,[0]] = x
    for k in range(N):
        u = np.dot(A,Q[:,k])
        for j in range(k+1):
            H[j,k] = np.dot(Q[:,j].conj().T,u)
            u = u-H[j,k]*Q[:,j]
        norm_u = np.linalg.norm(u)
        if k < N-1:
            H[k+1,k] = norm_u
            if norm_u == 0:
                break
            Q[:,[k+1]] = u.reshape(n,1)/norm_u
    H = H[:N,:N]
    return H

def Lanczos(A,N):
    '''
    Lanczos迭代
    Parameters:
        A:对称或Hermite矩阵
        N:Kroylov子空间维数
    Return:
        H:三对角矩阵
    '''
    n = A.shape[0]
    Q = np.zeros((n,N))
    alpha = np.zeros(N)
    beta = np.zeros(N)
    x = np.ones((n,1))/n**0.5 #初始向量
    Q[:,[0]] = x
    for k in range(N):
        u = np.dot(A,Q[:,k])
        alpha[k] = np.dot(Q[:,k].conj().T,u)
        u = u-beta[k-1]*Q[:,k-1]-alpha[k]*Q[:,k]
        norm_u = np.linalg.norm(u)
        if k < N-1:
            beta[k+1] = norm_u
            if norm_u == 0:
                break
            Q[:,[k+1]] = u.reshape(n,1)/norm_u
    H = np.diag(alpha)+np.diag(beta[1:],k=1)+np.diag(beta[1:],k=-1)
    return H
    
def QR(A,tol=1e-7):
    '''
    QR迭代求矩阵特征值
    Parameters:
        A:任意矩阵
        tol:精度
    Return:
        evalue：A的特征值
    '''
    eps = 1 #误差
    while eps>tol:
        Q,R = np.linalg.qr(A) 
        d0 = np.diag(A)
        A = np.dot(R,Q)
        d1 = np.diag(A) 
        eps = np.max(abs(d0-d1)) 
    evalue = d1
    return evalue


n = 25
a = np.ones(n)

########测试QR函数
A = np.diag(5*a)+np.diag(2*a[1:],k=1)+np.diag(a[1:],k=-1)
eig = QR(A)
w2,v= np.linalg.eig(A)
w2 = w2[w2.argsort()[::-1]]
eig = eig[eig.argsort()[::-1]]
print(abs(np.max(abs(eig-w2))) )
  
########测试Arnoldi函数
err = np.zeros(16)
for N in range(5,21):
    H = Arnoldi(A,N)
    w1,v = np.linalg.eig(H)
    w1 = w1[w1.argsort()[::-1]]
    err[N-5] = np.max(abs(w1[:5]-w2[:5]))
   
x = np.arange(5,21,1)
plt.figure(1)
plt.plot(x, err)
plt.xlabel('Kroylov子空间维数')  
plt.ylabel('Arnoldi') 
plt.show()
print(H)

  
########测试Lanczos函数
A = np.diag(2*a)+np.diag(a[1:],k=1)+np.diag(a[1:],k=-1)
w2,v= np.linalg.eig(A)
w2 = w2[w2.argsort()[::-1]]
   
err = np.zeros(16)
for N in range(5,21):
    H = Lanczos(A,N)
    w1,v = np.linalg.eig(H)
    w1 = w1[w1.argsort()[::-1]]
    err[N-5] = np.max(abs(w1[:5]-w2[:5]))
plt.figure(2)
x = np.arange(5,21,1)
plt.plot(x, err)
plt.xlabel('Kroylov子空间维数')  
plt.ylabel('Lanczos') 
plt.show()
print(H)