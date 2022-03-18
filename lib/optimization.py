#
# Iterative optimization algorithm
#
from math import *
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

from performance import calc_la, calc_lb
from bindings import generate_sparse_Kcomp

# Random initialization of P, Q, R
# variance is set to 1/(N*Nc)
#
def initialize_PQR_random(N, Nc):
    P = nrnd.normal(0.0, 1.0/sqrt(float(N*Nc)), (N,N,Nc))
    Q = nrnd.normal(0.0, 1.0/sqrt(float(N*Nc)), (N,N,Nc))
    R = nrnd.normal(0.0, 1.0/sqrt(float(N*Nc)), (N,N,Nc))
    return P, Q, R

# An update of P by Q
# Due to symmetry, an update of Q by P is done with update_PbyQ(Q, P)
#
def update_PbyQ(P, Q):
    N = len(P); Nc = len(P[0,0])
    
    gmQ1tmp = np.tensordot(Q, Q, axes=([0,1],[0,1]))
    gmQ1 = np.transpose(np.tensordot(np.identity(N), gmQ1tmp, axes=0), axes=[0,2,1,3])
    
    gmQ2 = np.tensordot(Q, Q, axes=([0],[0]))
    gmQ3 = np.transpose(gmQ2, axes=[0,3,2,1])
    gmQ = np.reshape(gmQ1 + gmQ2 + gmQ3, [N*Nc, N*Nc])

    Qvec = np.reshape(Q, [N, N*Nc])
    newPvec = np.zeros((N,N*Nc))
    for n in range(N):
        newPvec[n] = nlg.solve(gmQ, Qvec[n])

    return np.reshape(newPvec, [N,N,Nc])

# An update of P by R
def update_PbyR(P, R):
    Ptick = np.transpose(P, axes=[1,0,2])
    Rtick = np.transpose(R, axes=[1,0,2])
    
    newPtick = update_PbyQ(Ptick, Rtick)
    return np.transpose(newPtick, axes=[1,0,2])

#iterative optimization of P, Q, R
def optimize_PQR_rand(N, Nc, Tpmax):
    P, Q, R = initialize_PQR_random(N, Nc)
    errs = np.zeros((2,Tpmax))
    for t in range(Tpmax):
        Q = update_PbyQ(Q, P)
        P = update_PbyQ(P, Q)
        R = update_PbyR(R, P)
        P = update_PbyR(P, R)
        errs[0,t] = calc_la(P, Q)
        errs[1,t] = calc_lb(P, R)
        print(t, errs[0,t]/float(N), errs[1,t]/float(N))

    return P, Q, R, errs

#Optimization from a small perturbation to one of K-compositional binding
def optimize_PQR_Kcomp(N, Nc, Tpmax, K, sigr):
    Pzero = generate_sparse_Kcomp(N, K)
    sig = sigr/sqrt( float(N*Nc) )
    P = Pzero + nrnd.normal( 0, sig, (N,N,Nc) )
    Q = Pzero + nrnd.normal( 0, sig, (N,N,Nc) )
    R = Pzero + nrnd.normal( 0, sig, (N,N,Nc) )

    errs = np.zeros((2,Tpmax))
    for t in range(Tpmax):
        Q = update_PbyQ(Q, P)
        P = update_PbyQ(P, Q)
        R = update_PbyR(R, P)
        P = update_PbyR(P, R)
        errs[0,t] = calc_la(P, Q)
        errs[1,t] = calc_lb(P, R)
        print(t, errs[0,t]/float(N), errs[1,t]/float(N))
    
    return P, Q, R, errs

#iterative optimization of P, Q, R, with norm tracking
def optimize_PQR_dPQR(N, Nc, Tpmax):
    P, Q, R = initialize_PQR_random(N, Nc)
    errs = np.zeros((2,Tpmax))
    norms = np.zeros((2,Tpmax))
    syms = np.zeros((2,Tpmax))
    for t in range(Tpmax):
        errs[0,t] = calc_la(P, Q)
        errs[1,t] = calc_lb(P, R)
        
        dQP = Q - (np.amax(Q)/np.amax(P))*P
        dQR = Q - R
        
        norms[0,t] = np.sum( np.multiply(dQP, dQP) )
        norms[1,t] = np.sum( np.multiply(dQR, dQR) )
        
        for i in range(N):
            PQti = np.dot(P[i,:,:], np.transpose(Q[i,:,:]))
            dPQti = PQti - np.transpose(PQti)
            syms[0,t] += np.sum( np.multiply(dPQti, dPQti) )
            
            PRti = np.dot(P[:,i,:], np.transpose(R[:,i,:]))
            dPRti = PRti - np.transpose(PRti)
            syms[1,t] += np.sum( np.multiply(dPRti, dPRti) )
    
        #print(t, errs[0,t]/float(N), errs[1,t]/float(N))
        Q = update_PbyQ(Q, P)
        P = update_PbyQ(P, Q)
        R = update_PbyR(R, P)
        P = update_PbyR(P, R)

    return P, Q, R, errs, norms, syms



