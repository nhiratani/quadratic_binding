#
# Unbinding with dictionary
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
from performance import calc_dictionary_errs

def simul(N, d, L, D, Tmax):
    K = 8 #Octonions
    Poct = generate_ext_sKcomp(N, K, d)
    Pth = generate_tensor_HRR(N, d)
    Prand = generate_random_quad(N, d)
    
    festr = 'data/qbind_dict_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_L' + str(L) + '_D' + str(D) + '_Tm' + str(Tmax) + '.txt'
    fwe = open(festr,'w')
    
    for q in range(3):
        if q == 0:
            errtmps = calc_dictionary_errs(Poct, Poct, Poct, L, D, Tmax)
        elif q == 1:
            errtmps = calc_dictionary_errs(Pth, Pth, Pth, L, D, Tmax)
        else:
            errtmps = calc_dictionary_errs(Prand, Prand, Prand, L, D, Tmax)
        fwetmp = str(D)
        for widx in range(7):
            fwetmp += " " + str(errtmps[widx])
        fwe.write( fwetmp + "\n" )
    fwe.flush()

if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    d = int(param[2]) #expansion ratio
    L = int(param[3]) #bound pairs
    D = int(param[4]) #dictionary size
    Tmax = int(param[5])
    simul(N, d, L, D, Tmax)

