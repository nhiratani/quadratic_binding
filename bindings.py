#
# Binding algorithms
#
from math import *
import numpy as np
from numpy import random as nrnd
#from numpy import linalg as nlg
#from scipy import stats as scist

import cayley_dickson as KD

#####################
## K-compositional ##
#####################

#transform a KD object into a numpy array
def KD_to_numpy(M):
    Mlen = len(M)
    lnM = int( log2(Mlen) )
    
    Mp = np.zeros((Mlen,Mlen,Mlen))
    for i in range(Mlen):
        for j in range(Mlen):
            mtmps = []
            for l in range(lnM):
                mtmps.append([])
                if l == 0:
                    mtmps[l].append( M[i,j].a )
                    mtmps[l].append( M[i,j].b )
                else:
                    for midx in range(len(mtmps[l-1])):
                        mtmps[l].append( mtmps[l-1][midx].a )
                        mtmps[l].append( mtmps[l-1][midx].b )
            for k in range(Mlen):
                Mp[i,j,k] = mtmps[lnM-1][k]
    return Mp

#generate a tensor consist of a solution of the Hurwitz matrix equations
def generate_Hurwitz_mats(K):
    lnK = int(log2(K))
    
    real_basis = [1]
    for q in range(lnK):
        if q == 0:
            Htsr = KD.KD_construction(real_basis)
        else:
            Htsr = KD.KD_construction(Htsr.index)
    return KD_to_numpy( Htsr.to_numpy() )

#sparse K-compositional
def generate_sparse_Kcomp(N, K):
    q = int(N/K)
    lmbd = 1.0/sqrt(K + 2.0) #normalization factor
    
    Htsr = generate_Hurwitz_mats(K)
    sKcomp = np.zeros((N,N,N))

    for i in range(q):
        for j in range(K):
            ntmp = i*K + j
            sKcomp[ntmp,i*K:(i+1)*K,i*K:(i+1)*K] = lmbd*Htsr[j,:,:]
    return sKcomp

#Extended sparse K-compositional
def generate_ext_sKcomp(N, K, d):
    Nd = int(N*d)
    q = int(N/K)
    lmbd = 1.0/sqrt( float(d*K) ) #normalization factor
    
    Htsr = generate_Hurwitz_mats(K)
    ext_sKcomp = np.zeros((N,N,Nd))
    
    for i in range(N):
        for m in range(d):
            mu = int(floor( i/q ))
            nu = (i+m)%q
            ext_sKcomp[i,nu*K:(nu+1)*K,m*N+nu*K:m*N+(nu+1)*K] = lmbd*Htsr[mu,:,:]
    #for n in range(N):
    #    print(sKcomp[n,:,:])
    return ext_sKcomp

#Hadamard binding
def generate_hadamard(N):
    P = np.zeros((N,N,N))
    for i in range(N):
        P[i,i,i] = 1.0/sqrt(1.0 + 2.0)
    return P

################
## Tensor-HRR ##
################

# Tensor-HRR binding
# m = 1: HRR (Holographic reduced representation)
# m = N: tensor productrepresentation
def generate_tensor_HRR(N, d):
    Phad = np.zeros((N,N,d*N))
    for i in range(N):
        for j in range(N):
            Phad[i, j, (i*d+j)%(d*N)] = 1
    return (1.0/sqrt(N))*Phad

#normalized HRR binding
def generate_norm_HRR(N):
    P = np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):
            P[i,j, (i+j)%N] = 1
    return (1.0/sqrt(2.0*(N+1)))*P

############
## random ##
############

#random quadratic binding (Nc = N*m)
def generate_random_quad(N, m):
    Nm = int(N*m)
    gen_gauss = np.zeros((N,N,Nm))
    for i in range(N):
        gen_gauss[i,:,:] = nrnd.normal(0.0, 1.0/np.sqrt(float(N)), (N,Nm))
    return ( 1.0/sqrt(N*m) )*gen_gauss

