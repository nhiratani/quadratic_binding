#
# Sparse K-compositional binding
# and holographic reduced representation (HRR)
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

from bindings import generate_sparse_Kcomp, generate_hadamard, generate_norm_HRR
from performance import calc_decoding_errs

def simul(N, K, Tmax):
    if K > 1:
        P = generate_sparse_Kcomp(N, K)
    elif K == 1:
        P = generate_hadamard(N)
    elif K == 0:
        P = generate_norm_HRR(N)

    festr = 'data/qbind_sKcomp_HRR_error_N' + str(N) + '_K' + str(K) + '_Tm' + str(Tmax) + '.txt'
    fwe = open(festr,'w')

    errtmps = calc_decoding_errs(P, P, P, 1, Tmax)
    fwe.write( str(errtmps[0]) + " " + str(errtmps[1]) + " " + str(errtmps[2]) + " " + str(errtmps[3]) + "\n" )
    fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    K = int(param[2])
    Tmax = int(param[3])
    simul(N, K, Tmax)

