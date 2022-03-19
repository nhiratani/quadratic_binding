#
# Iterative numerical optimization of P, Q, R
# from sparse K-compositional tensors
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

from optimization import optimize_PQR_Kcomp

def simul(N, Tpmax, K, sigr, ik):
    P, Q, R, errs = optimize_PQR_Kcomp(N, N, Tpmax, K, sigr)
    
    festr = 'data/qbind_Kcomp_basin_errs_N' + str(N) + '_Tpm' + str(Tpmax) + '_K' + str(K) + '_sigr' + str(sigr) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')
    for t in range(Tpmax):
        fwe.write( str(t) + " " + str(errs[0,t]) + " " + str(errs[1,t]) + "\n" )

if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    Tpmax = int(param[2])
    K = int(param[3])
    sigr = float(param[4])
    ik = int(param[5])
    simul(N, Tpmax, K, sigr, ik)

