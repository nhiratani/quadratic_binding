#
# Iterative optimization algorithm
#
# Under L >= 1
#
from math import *
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

# Random initialization of P, Q, R
# variance is set to 1/(N*Nc)
#
def initialize_PQR_random(N, Nc):
    P = nrnd.normal(0.0, 1.0/sqrt(float(N*Nc)), (N,N,Nc))
    Q = nrnd.normal(0.0, 1.0/sqrt(float(N*Nc)), (N,N,Nc))
    R = nrnd.normal(0.0, 1.0/sqrt(float(N*Nc)), (N,N,Nc))
    return P, Q, R

#
# Evaluation of the losses la and lb under L >= 1
#
def calc_la_L(P, Q, L):
    N = len(P)
    
    la1 = N
    la2 = -2.0*np.sum( np.multiply(P, Q) )
    
    PQli = np.tensordot(P, Q, axes=([1,2], [1,2]))
    la3 = np.sum( np.multiply(PQli, PQli) )
    
    PQlmij = np.tensordot(P, Q, axes=([2],[2]))
    la4 = np.sum( np.multiply(PQlmij, PQlmij) )
    
    PQljim = np.transpose( PQlmij, axes=[0,3,2,1] )
    la5 = np.sum( np.multiply(PQlmij, PQljim) )
    
    return la1 + la2 + la3 + L*la4 + la5

def calc_lb_L(P, R, L):
    Ptick = np.transpose(P, axes=[1,0,2])
    Rtick = np.transpose(R, axes=[1,0,2])
    return calc_la_L(Ptick, Rtick, L)

# An update of P by Q
# Due to symmetry, an update of Q by P is done with update_PbyQ(Q, P)
#
def update_PbyQ_L(P, Q, L):
    N = len(P); Nc = len(P[0,0])
    
    gmQ1tmp = np.tensordot(Q, Q, axes=([0,1],[0,1]))
    gmQ1 = np.transpose(np.tensordot(np.identity(N), gmQ1tmp, axes=0), axes=[0,2,1,3])
    
    gmQ2 = np.tensordot(Q, Q, axes=([0],[0]))
    gmQ3 = np.transpose(gmQ2, axes=[0,3,2,1])
    gmQ = np.reshape(L*gmQ1 + gmQ2 + gmQ3, [N*Nc, N*Nc])
    
    Qvec = np.reshape(Q, [N, N*Nc])
    newPvec = np.zeros((N,N*Nc))
    for n in range(N):
        newPvec[n] = nlg.solve(gmQ, Qvec[n])
    
    return np.reshape(newPvec, [N,N,Nc])

# An update of P by R
def update_PbyR_L(P, R, L):
    Ptick = np.transpose(P, axes=[1,0,2])
    Rtick = np.transpose(R, axes=[1,0,2])
    
    newPtick = update_PbyQ_L(Ptick, Rtick, L)
    return np.transpose(newPtick, axes=[1,0,2])

#iterative optimization of P, Q, R, with norm tracking
def optimize_PQR_dPQR_L(N, Nc, L, Tpmax):
    P, Q, R = initialize_PQR_random(N, Nc)
    errs = np.zeros((2,Tpmax))
    norms = np.zeros((2,Tpmax))
    syms = np.zeros((2,Tpmax))
    for t in range(Tpmax):
        errs[0,t] = calc_la_L(P, Q, L)
        errs[1,t] = calc_lb_L(P, R, L)
        
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
        Q = update_PbyQ_L(Q, P, L)
        P = update_PbyQ_L(P, Q, L)
        R = update_PbyR_L(R, P, L)
        P = update_PbyR_L(P, R, L)
    
    return P, Q, R, errs, norms, syms
