#
# Readout of unbinding performance
#

from math import *
import numpy as np
from scipy import stats as scist

import matplotlib.pyplot as plt

clrs = ['C1', 'C0', 'C3'] #(ext-Oct, Tensor-HRR, Random-quad)

N = 128
ds = range(1,11)
Tmax = 10000

Ls = range(1,11)
dlen = len(ds); Llen = len(Ls)
errs = np.zeros((3, Llen, dlen)) #(ext-Oct, Tensor-HRR, Random-quad)

for didx in range(dlen):
    d = ds[didx]
    festr = 'data/qbind_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_Tm' + str(Tmax) + '.txt'
    lidx = 0
    for line in open(festr, 'r'):
        ltmp = line[:-1].split(" ")
        for widx in range(3):
            errs[widx, lidx, didx] = float(ltmp[1+widx])
        lidx += 1

errs_theory = np.zeros((3, Llen, dlen))
for didx in range(dlen):
    d = ds[didx]; Nc = N*d
    for lidx in range(Llen):
        L = Ls[lidx]
        #Extended Octonions
        K = 8
        errs_theory[0, lidx, didx] = N*N*(L - 1 + 2.0/K)/float(N*Nc)

        #Tensor-HRR
        errs_theory[1, lidx, didx] = 1/float(N) + L*N*N/float(N*Nc)

        #random-quadratic
        errs_theory[2, lidx, didx] = L*(N*N + Nc + 1)/float(N*Nc) + (Nc + 3*N + 1)/float(N*Nc)

loidxs = [3, 10]
for lo in range(2):
    svfg = plt.figure()
    loidx = loidxs[lo]
    
    for widx in range(3):
        plt.plot(ds, errs_theory[widx, loidx-1], '-', color=clrs[widx])
        plt.plot(ds, errs[widx, loidx-1], 'o', color=clrs[widx], ms=5)

    if loidx == 3:
        plt.ylim(0.0, 3.3)
        plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=16)
    elif loidx == 10:
        plt.ylim(0.0, 11)
        plt.yticks([0, 2, 4, 6, 8, 10], fontsize=16)

    plt.xticks(fontsize=16)
    plt.show()
    svfg.savefig('fig_qbind_ext_error_readout1_d1-10' + '_L' + str(loidx) + '_N' + str(N) + '_Tm' + str(Tmax) + '.pdf')



