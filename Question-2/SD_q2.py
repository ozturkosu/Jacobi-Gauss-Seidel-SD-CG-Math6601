import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
import matplotlib.pyplot as plt


def SD(b, R , N , x ,h )  :


    #x= np.zeros((N+2,N+2))
    m= 0
    tol = 1

    Ar = np.zeros((N+2,N+2))

    alpha=0
    part1=0
    part2=0

    error =[]
    Iteration=[]


    while tol >= 10**-10:


        if m == 0:
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

            normfirst=linalg.norm(R,'fro')
        #Calculate Ar
        for i in range(N+2):
            for j in range(N+2):
                if i==0:
                    Ar[i][j] = 0
                elif i==N+1:
                    Ar[i][j] = 0
                elif j==0:
                    Ar[i][j] = 0
                elif j==N+1:
                    Ar[i][j] = 0
                else:
                    Ar[i][j] = (4 + h**2)*R[i][j] - R[i-1][j] - R[i][j-1] - R[i+1][j] - R[i][j+1]

        #alpha = np.dot(np.transpose(R), R) / np.dot(np.transpose(R) , Ar )

        for j in range(1,N+1):
            for i in range(1,N+1):
                part1 = part1 + (R[i][j]*R[i][j]) #/(R[i][j]*Ar[i][j])

        for j in range(1,N+1):
            for i in range(1,N+1):
                part2 = part2 + (R[i][j]*Ar[i][j])

        alpha= part1/part2

        #x = x + alpha*R
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
                else:
                    x[i][j] = x[i][j] + alpha* R[i][j]

        # r = b - Ax
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




        print (m)
        norm=linalg.norm(R, 'fro')
        tol = norm / normfirst
        error.append(np.log10(norm))
        print(tol)
        m = m + 1
        Iteration.append(m)

    print (m)
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
            b[i][j] = (1/(N+1))**2

R = np.zeros((N+2,N+2))
for i in range(N+2):
    for j in range(N+2):
        R[i][j] = b[i][j]

print("Before solution")

h=1/(N+1)
sol = SD(b , R , N , x, h)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
