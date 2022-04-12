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

#readout of Oct, HRR, Random bindings
for didx in range(dlen):
    d = ds[didx]
    festr = 'data/qbind_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_Tm' + str(Tmax) + '.txt'
    lidx = 0
    for line in open(festr, 'r'):
        ltmp = line[:-1].split(" ")
        for widx in range(3):
            errs[widx, lidx, didx] = float(ltmp[1+widx])
        lidx += 1

#readout of sum binding performance
sum_errs = np.zeros((Llen))
for lidx in range(Llen):
    L = Ls[lidx]
    festr = 'data/qbind_sum_bind_error_N' + str(N) + '_L' + str(L) + '_Tm' + str(Tmax) + '.txt'
    for line in open(festr, 'r'):
        ltmp = line[:-1].split(" ")
        sum_errs[lidx] = float(ltmp[0])

Ltheos = range(1,15,1)
errs_theory = np.zeros((3, len(Ltheos), dlen))
for didx in range(dlen):
    d = ds[didx]; Nc = N*d
    for lidx in range( len(Ltheos) ):
        L = Ltheos[lidx]
        #Extended Octonions
        K = 8
        errs_theory[0, lidx, didx] = N*N*(L - 1 + 2.0/K)/float(N*Nc)

        #Tensor-HRR
        errs_theory[1, lidx, didx] = 1/float(N) + L*N*N/float(N*Nc)

svfg = plt.figure()

plt.plot(Ls, sum_errs, 'o', color='C2', ms=5)
plt.plot(Ls, 2.0*np.array(Ls) - 2.0*np.ones((Llen)), '-', color='C2' )
for widx in range(2):
    plt.plot(Ltheos, errs_theory[widx, :, 0], '-', color=clrs[widx])
    plt.plot(Ls, errs[widx, :, 0], 'o', color=clrs[widx], ms=5)

plt.ylim(0.0, 1.1*errs[2,Llen-1,0])
plt.xlim(0, 10.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
svfg.savefig('fig_qbind_Oct_HRR_Sum_readout6_L1-10' + '_d1' + '_N' + str(N) + '_Tm' + str(Tmax) + '.pdf')



