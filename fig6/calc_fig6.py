#
# Iterative numerical optimization of P, Q, R
# under Nc = N and L >= 1
#
### For parallel processing
import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'

from math import *
import sys
import numpy as np

from ext_optimization import optimize_PQR_dPQR_L

def simul(N, Tpmax, L, ik):
    P, Q, R, errs, norms, syms = optimize_PQR_dPQR_L(N, N, L, Tpmax)
    
    festr = 'data/qbind_opt_PQR_err_N' + str(N) + '_L' + str(L) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')
    for t in range(Tpmax):
        fwe.write( str(t) + " " + str(errs[0,t]) + " " + str(errs[1,t]) + " " + str(norms[0,t]) + " " + str(norms[1,t]) + " " + str(syms[0,t]) + " " + str(syms[1,t]) + "\n" )

    if ik == 0 and N <= 64: #record P, Q, R only for a small N
        fpstr = 'data/qbind_opt_PQR_L_tensor_N' + str(N) + '_L' + str(L) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
        fwp = open(fpstr,'w')
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    fwp.write( str(P[i,j,k]) + " " + str(Q[i,j,k]) + " " + str(R[i,j,k]) + "\n" )

if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    Tpmax = int(param[2])
    L = int(param[3])
    ik = int(param[4])
    simul(N, Tpmax, L, ik)

