#
# Iterative numerical optimization of P, Q, R
# under Nc = N
#
### For parallel processing
import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'

from math import *
import sys
import numpy as np
sys.path.append('../lib')

from optimization import optimize_PQR_dPQR

def simul(N, Tpmax, ik):
    P, Q, R, errs, norms, syms = optimize_PQR_dPQR(N, N, Tpmax)
    
    festr = 'data/qbind_opt_PQR_err_N' + str(N) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')
    for t in range(Tpmax):
        fwe.write( str(t) + " " + str(errs[0,t]) + " " + str(errs[1,t]) + " " + str(norms[0,t]) + " " + str(norms[1,t]) + " " + str(syms[0,t]) + " " + str(syms[1,t]) + "\n" )


if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    Tpmax = int(param[2])
    ik = int(param[3])
    simul(N, Tpmax, ik)

