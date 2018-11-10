import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
import matplotlib.pyplot as plt

def gaussseidel( b ,R, N , x=None ):

  # Create an initial guess if needed
    if x is None:
      x = zeros(N)


    # Create a vector of the diagonal elements of A
    # and subtract them from A
    #D = diag(A)
    #R = A - diagflat(D)

    x_previous=np.zeros(N)
    k=0 ;

    normfirst= linalg.norm(R)

    error =[]
    Iteration=[]

    for  j in range(100000):

        for m in range(N) :
            x_previous[m]= x[m]

        norm=linalg.norm(R)
        print(norm / normfirst)

        if (norm / normfirst) < 10**-10:
            break;

        error.append(np.log10(norm))
        k= k+1
        #print(k)

        Iteration.append(k)

        for i in range(N):
            if i==0:
                x[i] = (b[i] +  x_previous[i+1])/2
            elif  i == N-1 :
                x[i] = (b[i] +  x[i-1])/2
            else :
                x[i] = (b[i]+  x[i-1] + x_previous[i+1])/2

        for i in range(N):
            if i==0:
                R[i] = b[i] + x[i+1] -2*x[i]
            elif  i == N-1:
                R[i] = b[i] + x[i-1] -2*x[i]
            else  :
                R[i] = b[i] + x[i-1] -2*x[i] + x[i+1]




    print (k);
    plt.plot(Iteration,error)
    plt.title(' Residual VS Iteration')
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
sol = gaussseidel(b , R , N , x)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
