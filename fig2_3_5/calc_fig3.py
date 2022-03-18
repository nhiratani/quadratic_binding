#
# Iterative numerical optimization of P, Q, R
# under Nc = N
#
# Recording the elements of P, Q, R
#
# Caution: output file size is large (5-50MB)
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

from optimization import optimize_PQR_rand

#import matplotlib.pyplot as plt
#from pylab import cm

def simul(N, Tpmax, ik):
    P, Q, R, errs = optimize_PQR_rand(N, N, Tpmax)
    
    #festr = 'data/qbind_opt_PQR_err_N' + str(N) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
    #fwe = open(festr,'w')
    #for t in range(Tpmax):
    #    fwe.write( str(t) + " " + str(errs[0,t]) + " " + str(errs[1,t]) + "\n" )

    if ik < 3: #record P, Q, R only for some random seeds
        fpstr = 'data/qbind_opt_PQR_tensor_N' + str(N) + '_Tpm' + str(Tpmax) + '_ik' + str(ik) + '.txt'
        fwp = open(fpstr,'w')

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    fwp.write( str(P[i,j,k]) + " " + str(Q[i,j,k]) + " " + str(R[i,j,k]) + "\n" )



if __name__ == "__main__":
    param = sys.argv
    N = int(param[1])
    Tpmax = int(param[2])
    ik = int(param[3])
    simul(N, Tpmax, ik)

