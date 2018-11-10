import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
import matplotlib.pyplot as plt

def CG( R , N , x , h )  :


    x= np.zeros((N+2, N+2))
    m= 0
    tol = 1

    q = np.zeros((N+2, N+2)) #q
    p = np.zeros((N+2, N+2))
    R_temp=np.zeros((N,N))

    alpha=0
    beta=0

    v=0
    v_plus=0
    Mu=0

    error =[]
    Iteration=[]

    partv=0
    partvp=0
    partMu=0

    while tol >= 10**-10:

        if m == 0 :
            for i in range(N+2):
                for j in range(N+2):
                    if i==0:
                        R[i][j]=R[i][j]
                    elif i==N+1:
                        R[i][j]=R[i][j]
                    elif j==0:
                        R[i][j]=R[i][j]
                    elif j==N+1:
                        R[i][j]=R[i][j]
                    else:
                        R[i][j]= R[i][j]-((4 + h**2)*x[i][j] - x[i-1][j] - x[i][j-1] - x[i+1][j] - x[i][j+1])

            #for i in range(1,N):
            #    for j in range(1,N):
            #        R_temp[i-1][j-1]=R[i][j]


            normfirst=linalg.norm(R,'fro')

            for i in range(N+2):
                for j in range(N+2):
                        p[i][j] = R[i][j]

            for j in range(1,N+1):
                for i in range(1,N+1):
                    partv = partv + (R[i][j]*R[i][j])
            v=partv
            partv=0
        else:

            #Calculate q= Ap
            for i in range(N+2):
                for j in range(N+2):
                    if i==0:
                        q[i][j] = 0
                    elif i==N+1:
                        q[i][j] = 0
                    elif j==0:
                        q[i][j] = 0
                    elif j==N+1:
                        q[i][j] = 0
                    else:
                        q[i][j] = (4 + h**2)*p[i][j] - p[i-1][j] - p[i][j-1] - p[i+1][j] - p[i][j+1]



            #Calculate Mu= p^T q
            for j in range(1,N+1):
                for i in range(1,N+1):
                    partMu = partMu + (p[i][j]*q[i][j])

            Mu=partMu
            partMu=0
            alpha= v/Mu



            #x = x + alpha* p
            for i in range(N+2):
                for j in range(N+2):
                        x[i][j] = x[i][j] + alpha* p[i][j]




            #R = R - alpha * Ap
            for i in range(N+2):
                for j in range(N+2):
                        R[i][j] = R[i][j] -alpha*q[i][j]




            #v_plus= r^Tr
            for j in range(1,N+1):
                for i in range(1,N+1):
                    partvp = partvp + (R[i][j]*R[i][j])



            v_plus=partvp
            partvp=0
            beta=v_plus / v

            #p = R+ beta*p

            for i in range(N+2):
                for j in range(N+2):
                    if i==0:
                        p[i][j] = 0
                    elif i==N+1:
                        p[i][j] = 0
                    elif j==0:
                        p[i][j] = 0
                    elif j==0:
                        p[i][j] = 0
                    else:
                        p[i][j] = R[i][j] + beta*p[i][j]

            #v=v_plus
            v=v_plus

            #for i in range(1,N):
            #    for j in range(1,N):
            #        R_temp[i-1][j-1]=R[i][j]
            norm=linalg.norm(R,'fro')
            error.append(np.log10(norm))
            tol = norm/ normfirst
            print(tol)

        m = m + 1
        Iteration.append(m)

    print (m);
    plt.plot(error)
    plt.ylabel('log10(||r^m||)')
    plt.xlabel('Iteration')
    plt.show()
    return x

N=16;
M=N+2
x = np.zeros((M,M))

b= np.zeros((N+2,N+2))

h=1/(N+1)

for i in range (N+2):
    for j in range(N+2):
        if   i==0:
            b[i][j] = 0
        elif i==N+1:
            b[i][j] = 0
        elif j==0:
            b[i][j] = 0
        elif j==N+1:
            b[i][j] =0
        else:
            b[i][j] = h**2

R = np.zeros((N+2,N+2))

for i in range(N+2):
    for j in range(N+2):
        R[i][j] = b[i][j]




sol = CG( R , N , x, h)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
