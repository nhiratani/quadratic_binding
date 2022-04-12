#
# Estimate the performance of the sum bindings
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

from performance import calc_sum_err

def simul(N, L, Tmax):
    errs = calc_sum_err(N,L,Tmax)

    festr = 'data/qbind_sum_bind_error_N' + str(N) + '_L' + str(L) + '_Tm' + str(Tmax) + '.txt'
    fwe = open(festr,'w')

    fwe.write( str(errs[0]) + " " + str(errs[1]) + "\n" )
    fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    L = int(param[2])
    Tmax = int(param[3])
    simul(N, L, Tmax)

