import numpy as np
import math   as mt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from scipy import linalg
import matplotlib.pyplot as plt

def CG(b, R , N , x  )  :

    x= np.zeros(N)
    m= 0
    tol = 1

    Ap = np.zeros(N)
    p = np.zeros(N)

    error =[]
    Iteration=[]

    v=0
    v_plus=0
    Mu=0

    partv=0
    partvp=0
    partMu=0

    while tol >= 10**-10:

        if m == 0 :
            for i in range(N):
                if i==0:
                    R[i] = b[i] - 2*x[i] + x[i+1]
                elif i==N-1 :
                    R[i] = b[i] - 2*x[i] + x[i-1]
                else:
                    R[i] = b[i] - 2*x[i] + x[i-1] + x[i+1]

            for i in range(N):
                p[i] = R[i]

            normfirst=linalg.norm(R)
            for i in range(N):
                partv = partv + R[i]*R[i]

            v=partv
            partv=0

        #Calculate Ap
        for i in range(N):
            if i==0:
                Ap[i] =  2*p[i] - p[i+1]
            elif i==N-1:
                Ap[i] =  2*p[i] - p[i-1]
            else:
                Ap[i] =  2*p[i] - p[i-1] - p[i+1]


        #alpha = np.dot(np.transpose(R), R) / np.dot(np.transpose(p) , Ap )

        #Calculate Mu

        for i in range(N):
            partMu = partMu + p[i]*Ap[i]

        Mu=partMu
        partMu=0

        alpha=v/Mu

        x = x + alpha* p

        # r = b - alpha*A*p
        #R = R - alpha * Ap

        for i in range(N):
            R[i] = R[i] - alpha*Ap[i]


        #beta = np.dot(-np.transpose(R),Ap) / np.dot(np.transpose(p), Ap)

        #v_plus
        for i in range(N):
            partvp = partvp + R[i]*R[i]

        v_plus = partvp
        partvp=0

        beta = v_plus / v

        #p = R+ beta*p

        for i in range(N):
            p[i] = R[i] + beta*p[i]

        v=v_plus

        norm = linalg.norm(R)
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
sol = CG(b , R , N , x)

print ("b:")
pprint(b)

print ("x:")
pprint(sol)
