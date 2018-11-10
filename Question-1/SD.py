import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
import matplotlib.pyplot as plt




def SD(b, R , N , x_s  )  :

    # Create an initial guess if needed
    if x_s is None:
        x_s = zeros(N)

    error =[]
    Iteration=[]

    x= np.zeros(N)
    m= 0
    tol = 1

    Ar = np.zeros(N)
    normfirst=linalg.norm(R)

    while tol >= 10**-10:

        if m == 0 :
            for i in range(N):
                if i==0:
                    R[i] = b[i] - 2*x[i] + x[i+1]
                elif i==N-1 :
                    R[i] = b[i] - 2*x[i] + x[i-1]
                else:
                    R[i] = b[i] - 2*x[i] + x[i-1] + x[i+1]



        #Calculate Ar
        for i in range(N):
            if i==0:
                Ar[i] =  2*R[i] - R[i+1]
            elif i==N-1:
                Ar[i] =  2*R[i] - R[i-1]
            else:
                Ar[i] =  2*R[i] - R[i-1] - R[i+1]


        alpha = np.dot(np.transpose(R), R) / np.dot(np.transpose(R) , Ar )
        x = x + alpha* R

        # r = b - Ax
        for i in range(N):
            if i==0:
                R[i] = b[i] - 2*x[i] + x[i+1]
            elif i==N-1:
                R[i] = b[i] - 2*x[i] + x[i-1]
            else:
                R[i] = b[i] - 2*x[i] + x[i-1] + x[i+1]

        norm=linalg.norm(R)
        tol = norm / normfirst
        error.append(np.log10(norm))
        print(tol)
        m = m + 1
        Iteration.append(m)
    print (m);
    plt.plot(Iteration,error)
    plt.title('Residual VS Iteration')
    plt.ylabel('log10(||r^m||)')
    plt.xlabel('Iteration')
    plt.show()
    return x

N=40;
x = np.zeros(N)

b= np.zeros(N)
for i in range (N):
        b[i] = 1/(N+1)**2;

R = np.zeros(N)
for i in range (N):
        R[i] = b[i]

print("Before solution")
sol = SD(b , R , N , x)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
