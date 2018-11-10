import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
from numpy import linalg as LA
import matplotlib.pyplot as plt

def GS(b, R , N, x, h) :


    x_previous=np.zeros((N+2,N+2))

    k=0 ;

    normfirst= LA.norm(R, 'fro')

    error =[]
    Iteration=[]

    for  j in range(100000):

        print(LA.norm(R, 'fro') / LA.norm(b, 'fro'))


        norm=LA.norm(R, 'fro')


        if (LA.norm(R, 'fro') / LA.norm(b, 'fro')) < 10**-10:
            break;

        error.append(np.log10(norm))

        k= k+1

        Iteration.append(k)

        print(j)
        for j in range(N+2):
            for i in range(N+2):
                if i==0:
                    x[i][j] = 0
                elif i==N+1:
                    x[i][j] = 0
                elif j==0 :
                    x[i][j] = 0
                elif j==N+1 :
                    x[i][j] = 0
                else :
                    x[i][j]= (x[i-1][j] + x[i][j-1] + x[i][j+1] + x[i+1][j] + h**2)/(4+h**2)

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


    print (k)
    plt.plot(Iteration,error)
    plt.title('Residual VS Iteration')
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
sol = GS(b , R , N , x, h)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
