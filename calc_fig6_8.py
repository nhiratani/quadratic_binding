#
# Extended sparse K-compositional binding,
# Tensor Holographic binding
# and random binding
# under Nc >= N and L >= 1
#
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

from bindings import generate_ext_sKcomp, generate_tensor_HRR, generate_random_quad
from performance import calc_decoding_errs

def simul(N, d, Tmax):
    K = 8 #Octonions
    Poct = generate_ext_sKcomp(N, K, d)
    Pth = generate_tensor_HRR(N, d)
    Prand = generate_random_quad(N, d)

    festr = 'data/qbind_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_Tm' + str(Tmax) + '.txt'
    fwe = open(festr,'w')

    for lidx in range(1,11):
        oct_errs = calc_decoding_errs(Poct, Poct, Poct, lidx, Tmax)
        th_errs = calc_decoding_errs(Pth, Pth, Pth, lidx, Tmax)
        rand_errs = calc_decoding_errs(Prand, Prand, Prand, lidx, Tmax)
        fwe.write( str(lidx) + " " + str(oct_errs[0]) + " " + str(th_errs[0]) + " " + str(rand_errs[0]) + "\n" )
        fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    d = int(param[2])
    Tmax = int(param[3])
    simul(N, d, Tmax)

