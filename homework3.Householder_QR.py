
import numpy as np

sig = lambda x : np.sign(x) if x != 0 else 1

#不选主元的QR分解
def Householder(A,b):
    n = A.shape[1]#获取列数
    R = A.copy()
    c = b.copy()
    for i in range(n):
        v = R[i:,i].copy()
        alpha = -sig(v[0])*np.sqrt(np.sum(v*v))
        v[0] = v[0]-alpha
        beta = np.sum(v*v)
        if beta == 0:
            continue
        for j in range(i,n):
            R[i:,j] = R[i:,j]-2*np.sum(v*R[i:,j])/beta*v
        c[i:,0] = c[i:,0]-2*np.sum(v*c[i:,0])/beta*v   
    return R,c

#选主元的QR分解
def PivotHouseholder(A,b):
    n = A.shape[1]#获取列数
    R = A.copy()
    c = b.copy()
    p = np.arange(n)#记录置换顺序
    for i in range(n):
        #选取欧氏范数最大的列与第i列交换
        cnorm = np.sum(R[:,i:]*R[:,i:],axis=0)
        k = [j for j in range(n-i) if cnorm[j]==np.max(cnorm)][0]+i
        R[:,[i,k]] = R[:,[k,i]]
        p[[i,k]] = p[[k,i]]
        v = R[i:,i].copy()
        alpha = -sig(v[0])*np.sqrt(np.sum(v*v))
        v[0] = v[0]-alpha
        beta = np.sum(v*v)
        if beta == 0:
            continue
        for j in range(i,n):
            R[i:,j] = R[i:,j]-2*np.sum(v*R[i:,j])/beta*v
        c[i:,0] = c[i:,0]-2*np.sum(v*c[i:,0])*v/beta    
    return R,c,p

#==============   main   ======================================

m = 21
n = 12
eps = 10**(-10)
t = np.linspace(0, 1,m).reshape(m,1)
A = np.zeros((m,n))
for i in range(n):
    A[:,[i]] = t**i
    
    
N = 10#测试次数
err1 = np.zeros((N,2))#QR
err2 = np.zeros((N,2))#内置QR
err3 = np.zeros((N,2))#Cholesky分解 法方程
err4 = np.zeros((N,2))#内置函数求解法方程


for k in range(10):
    x = np.random.randint(-5,5,size=(n,1))
    u = (2*np.random.rand(m,1).astype(np.float32)-1)*eps
    y = np.dot(A,x)-u

    #QR
    R,c = Householder(A,y)
    x1 = np.linalg.solve(R[:n,:],c[:n])
    err1[k,0] = np.sum((y-np.dot(A,x1))**2)
    err1[k,1] = np.max(abs(x-x1))
    
    #numpy内置函数QR分解
    Q,R = np.linalg.qr(A,mode='complete')
    R = R[:n,:]
    y2 = np.dot(Q.T,y)[:n]
    x2 = np.linalg.solve(R,y2)
    err2[k,0] = np.sum((y-np.dot(A,x2))**2)
    err2[k,1] = np.max(abs(x-x2))
    
# =============================================================================
#     #主元QR
#     R,c,p = PivokHouseholder(A,y)
#     z = np.linalg.solve(R[:n,:],c[:n])
#     prink('p=',p)
#     x = np.zeros_like(z)
#     for i in range(n):
#         x[i] = z[p[i]]
#     err = [np.sum((y-np.dot(A,x))**2),np.max(abs(x-x))]
# =============================================================================
    
    
    #法方程 A'Ax=A'y，numpy内置函数Cholesky分解
    L = np.linalg.cholesky(np.dot(A.T,A))
    z = np.linalg.solve(L,np.dot(A.T,y))
    x3 = np.linalg.solve(L.T,z)
    err3[k,0] = np.sum((y-np.dot(A,x3))**2)
    err3[k,1] = np.max(abs(x-x3))
    
    #numpy内置函数直接解A'Ax=A'y
    x4 = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,y))
    err4[k,0] = np.sum((y-np.dot(A,x4))**2)
    err4[k,1] = np.max(abs(x-x4))
    

   
print('err1=',np.sum(err1,axis=0)/N)
print('err2=',np.sum(err2,axis=0)/N)
print('err3=',np.sum(err3,axis=0)/N)
print('err4=',np.sum(err4,axis=0)/N)
