import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
from numpy import linalg as LA
import matplotlib.pyplot as plt

def jacobi(b, R , N, x, h) :


    error =[]
    Iteration=[]

    x_previous=np.zeros((N+2,N+2))

    k=0 ;

    normfirst= LA.norm(R, 'fro')
    print(normfirst)


    for  j in range(100000):

        for m in range(N+2) :
            for n in range(N+2):
                x_previous[m][n]= x[m][n]
        norm=linalg.norm(R,'fro')
        print(norm)
        print(LA.norm(R, 'fro') / normfirst)

        if (LA.norm(R, 'fro') / normfirst) < 10**-10:
            break;

        error.append(np.log10(norm))

        k= k+1

        Iteration.append(k)

        print(j)
        for i in range(N+2):
            for j in range(N+2):
                if i==0:
                    x[i][j] = 0
                elif i==N+1:
                    x[i][j] = 0
                elif j==0 :
                    x[i][j] = 0
                elif j==N+1 :
                    x[i][j] = 0
                else :
                    x[i][j]= (x_previous[i-1][j] + x_previous[i][j-1] + x_previous[i][j+1] + x_previous[i+1][j] + h**2)/(4+h**2)

        for i in range(N+2):
            for j in range(N+2):
                if i==0:
                    R[i][j] =b[i][j]
                elif i==N+1:
                    R[i][j] =b[i][j]
                elif j==0:
                    R[i][j] =b[i][j]
                elif j==N+1:
                    R[i][j] =b[i][j]
                else:
                    R[i][j]= h**2 - ((4+h**2)*x[i][j] - x[i-1][j] - x[i][j-1] - x[i+1][j] -x[i][j+1])


    print (k);

    plt.plot(Iteration,error)
    plt.ylabel('log10(||r^m||)')
    plt.xlabel('Iteration')
    plt.show()



    return x


N=32;
M=N+2
x = np.zeros((M,M))

b= np.zeros((N+2,N+2))
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
            b[i][j] = (1/(N+1))**2;

R = np.zeros((N+2,N+2))
for i in range(N+2):
    for j in range(N+2):
        R[i][j] = b[i][j]

print("Before solution")

h=1/(N+1)
sol = jacobi(b , R , N , x, h)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
