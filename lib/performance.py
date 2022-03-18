#
# Performance evaluation of binding operators
#
from math import *
#import sys
import numpy as np
from numpy import random as nrnd

def calc_decoding_errs(P, Q, R, L, T):
    N = len(P)
    
    a_err = 0.0; a_err_var = 0.0
    b_err = 0.0; b_err_var = 0.0
    for t in range(T):
        a = nrnd.normal(0.0, 1.0, (N, L))
        b = nrnd.normal(0.0, 1.0, (N, L))
        ab = np.einsum('ik,jk->ijk', a, b)
        a0 = a[:,0]; b0 = b[:,0]
        
        c = np.sum( np.tensordot(P, ab, axes=([0,1],[0,1])), axis=1 )
        b0c = np.outer(b0, c)
        ahat = np.tensordot(Q, b0c, axes=([1,2],[0,1]))
        
        a0c = np.outer(a0, c)
        bhat = np.tensordot(R, a0c, axes=([0,2],[0,1]))
        
        #MSE
        a_errtmp = np.sum(np.multiply(a0-ahat, a0-ahat))/float(N)
        a_err += a_errtmp/float(T)
        a_err_var += a_errtmp*a_errtmp/float(T)
    
        b_errtmp = np.sum(np.multiply(b0-bhat, b0-bhat))/float(N)
        b_err += b_errtmp/float(T)
        b_err_var += b_errtmp*b_errtmp/float(T)
    
    a_err_sig = np.sqrt( a_err_var - a_err*a_err )
    b_err_sig = np.sqrt( b_err_var - b_err*b_err )
    return a_err, a_err_sig, b_err, b_err_sig

#The expected loss la given P and Q
def calc_la(P, Q):
    N = len(P)

    la1 = N
    la2 = -2.0*np.sum( np.multiply(P, Q) )
    
    PQli = np.tensordot(P, Q, axes=([1,2], [1,2]))
    la3 = np.sum( np.multiply(PQli, PQli) )

    PQlmij = np.tensordot(P, Q, axes=([2],[2]))
    la4 = np.sum( np.multiply(PQlmij, PQlmij) )
    
    PQljim = np.transpose( PQlmij, axes=[0,3,2,1] )
    la5 = np.sum( np.multiply(PQlmij, PQljim) )

    return la1 + la2 + la3 + la4 + la5

#The expected loss lb given P and R
def calc_lb(P, R):
    Ptick = np.transpose(P, axes=[1,0,2])
    Rtick = np.transpose(R, axes=[1,0,2])
    return calc_la(Ptick, Rtick)

#error under the sum binding
def calc_sum_err(N,L,T):
    a_err = 0.0; a_err_var = 0.0; a_sign_err = 0.0
    for t in range(T):
        a = nrnd.normal(0.0, 1.0, (N, L))
        b = nrnd.normal(0.0, 1.0, (N, L))
        a0 = a[:,0]; b0 = b[:,0]
        
        Lones = np.ones((L))
        c = np.dot(a + b, Lones)
        ahat = c - b0
        
        #MSE
        errtmp = np.sum(np.multiply(a0-ahat, a0-ahat))/float(N)
        a_err += errtmp/float(T)
        a_err_var += errtmp*errtmp/float(T)
    
    a_err_sig = np.sqrt( a_err_var - a_err*a_err )
    return a_err, a_err_sig

#calculate the decoding error under the dictionary
def calc_dictionary_errs(P, Q, R, L, D, T):
    N = len(P)
    
    a_err = 0.0; a_err_var = 0.0; b_err = 0.0; b_err_var = 0.0
    da_var1 = 0.0; da_varL = 0.0; da_varD = 0.0
    for t in range(T):
        a_dict = nrnd.normal(0.0, 1.0, (N, D))
        a = a_dict[:,:L]
        b_dict = nrnd.normal(0.0, 1.0, (N, D))
        b = b_dict[:,:L]
        ab = np.einsum('ik,jk->ijk', a, b)
        a0 = a[:,0]; b0 = b[:,0]
        
        c = np.sum( np.tensordot(P, ab, axes=([0,1],[0,1])), axis=1 )
        b0c = np.outer(b0, c)
        ahat = np.tensordot(Q, b0c, axes=([1,2],[0,1]))
        a0c = np.outer(a0, c)
        bhat = np.tensordot(R, a0c, axes=([0,2],[0,1]))
        
        #with dictionary error
        da = np.dot(ahat, a_dict)
        a_errtmp = 0.0 if (np.argmax(da) == 0) else 1.0
        db = np.dot(bhat, b_dict)
        b_errtmp = 0.0 if (np.argmax(db) == 0) else 1.0
        
        a_err += a_errtmp/float(T)
        a_err_var += a_errtmp*a_errtmp/float(T)
    
        b_err += b_errtmp/float(T)
        b_err_var += b_errtmp*b_errtmp/float(T)
    
        da2 = np.multiply(da, da)
        da_var1 += (da2[0] - N*N)/float(T)
        da_varL += np.average(da2[1:L])/float(T)
        da_varD += np.average(da2[L:])/float(T)
    
    a_err_sig = np.sqrt( a_err_var - a_err*a_err )
    b_err_sig = np.sqrt( b_err_var - b_err*b_err )
    
    da_chi1 = da_varD - L*N
    da_chi2 = da_varL - da_varD
    da_chi3 = da_var1 - da_varL
    
    return a_err, a_err_sig, b_err, b_err_sig, da_chi1, da_chi2, da_chi3


