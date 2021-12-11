import numpy as np

#回代法解上三角方程组Ux=b
def Solve(U,b):
    n = len(b)
    x = np.zeros((n,1))
    x[-1] = b[-1]/U[-1,-1]
    for i in range(n-2,-1,-1):
        x[i] = (b[i]-np.dot(U[i,i+1:],x[i+1:]))/U[i,i]
    return x

#顺序高斯消去法
def SeqGauss(A):
    n = A.shape[0] #A的行数
    for i in range(n-1):
        for j in range(i+1,n):
            A[j,i:] = A[j,i:]-(A[j,i]/A[i,i])*A[i,i:]
    return A

#列主元高斯消去法
def PivotGauss(A):
    n = A.shape[0] #A的行数
    for i in range(n-1):
        #找出列主元
        k = [j for j in range(i,n) if A[j,i]==np.max(A[i:,i])][0]
        #交换第i行和列主元所在的行
        A[[i,k],:]=A[[k,i],:]
        for j in range(i+1,n):
            A[j,i:] = A[j,i:]-(A[j,i]/A[i,i])*A[i,i:]
    return A
        
if __name__ == '__main__':
      
    N = 1000 #方程规模
    t = 10 #测试次数
    errA = np.zeros((t,2))
    errB = np.zeros((t,2))
    errC = np.zeros((t,2))
    
    for i in range(10):
        
        #构造系数矩阵   
        A = np.random.rand(N,N)#随机矩阵（一般都是可逆的）
        B = np.dot(A,A.T)/100#对称正定矩阵
        C = A+np.diag(np.sum(abs(A),axis=1))#对角占优矩阵
        
        #构造真解和右端向量
        x = np.random.rand(N, 1)#随机生成真解
        a = np.dot(A,x)#根据真解构造右端向量
        b = np.dot(B,x)
        c = np.dot(C,x)
        
        #用选主元和不选主元的高斯消元法求解Ax=a，Bx=b，Cx=c
        A1 = SeqGauss(np.hstack((A,a)))
        A2 = PivotGauss(np.hstack((A,a)))
        errA[i,0] = np.max(abs(x-Solve(A1[:,:-1],A1[:,-1])))
        errA[i,1] = np.max(abs(x-Solve(A2[:,:-1],A2[:,-1])))
        
        B1 = SeqGauss(np.hstack((B,b)))
        B2 = PivotGauss(np.hstack((B,b)))
        errB[i,0] = np.max(abs(x-Solve(B1[:,:-1],B1[:,-1])))
        errB[i,1] = np.max(abs(x-Solve(B2[:,:-1],B2[:,-1])))
        
        C1 = SeqGauss(np.hstack((C,c)))
        C2 = PivotGauss(np.hstack((C,c)))
        errC[i,0] = np.max(abs(x-Solve(C1[:,:-1],C1[:,-1])))
        errC[i,1] = np.max(abs(x-Solve(C2[:,:-1],C2[:,-1])))
    
    print('errA=',np.sum(errA,axis=0)/t)
    print('errB=',np.sum(errB,axis=0)/t)
    print('errC=',np.sum(errC,axis=0)/t)
    




