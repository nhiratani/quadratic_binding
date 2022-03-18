#
# Readout of unbinding performance with a dictionary
#

from math import *
#import sys
import numpy as np
from scipy import stats as scist
from scipy import special as scisp
from scipy import integrate as integrate

import matplotlib.pyplot as plt

clrs = ['C1', 'C0', 'C3'] #(ext-Oct, Tensor-HRR, Random-quad)

Ns = [96]
d = 1
Ls = range(2,21,1)
D = 5000
Tmax = 10000

Nlen = len(Ns)
Llen = len(Ls)

L_theos = range(1,41,1)

svfg = plt.figure()
for Nidx in range(Nlen):
    N = Ns[Nidx]
    aerrs = np.zeros((3, Llen))
    for Lidx in range(Llen):
        L = Ls[Lidx]
        festr = 'data/qbind_dict_ext_sKcomp_tensor_HRR_random_error_N' + str(N) + '_d' + str(d) + '_L' + str(L) + '_D' + str(D) + '_Tm' + str(Tmax) + '.txt'
        
        lidx = 0
        for line in open(festr, 'r'):
            ltmp = line[:-1].split(" ")
            aerrs[lidx, Lidx] = float(ltmp[1])
            lidx += 1

    K = 8
    dsig2s = np.array([[2.0/K, 2.0/K, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

    errs_theor = np.zeros((3, len(L_theos)))
    for q in range(3):
        for Lidx in range(len(L_theos)):
            L = L_theos[Lidx]
            sig2s = np.zeros((3,3))
            sig2s[q,0] = (L + dsig2s[q,0] + dsig2s[q,1] + dsig2s[q,2])/float(N)
            sig2s[q,1] = (L + dsig2s[q,0] + dsig2s[q,1])/float(N)
            sig2s[q,2] = (L + dsig2s[q,0])/float(N)
            m = 1.0/sqrt( sig2s[q,0] )
            
            errs_theor[q,Lidx] = 1.0 - integrate.quad( lambda x: ( ( 0.5*scisp.erfc(-x*sqrt(sig2s[q,0]/(2.0*sig2s[q,1]))) )**(L-1) )*( ( 0.5*scisp.erfc(-x*sqrt(sig2s[q,0]/(2.0*sig2s[q,2]))) )**(D-L) )*exp(-(x-m)*(x-m)/2.0)/sqrt(2.0*pi), -np.inf, np.inf )[0]

    for widx in range(3):
        if widx < 2:
            plt.plot(L_theos, errs_theor[widx], '-', color=clrs[widx])
    for widx in [0,1,2]:
        plt.plot(Ls, aerrs[widx], 'o', color=clrs[widx], ms=7.5)

plt.xlim(0, 15)
plt.ylim(0.0, 1.0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()
svfg.savefig('fig_qbind_dict_ext_readout7a_D' + str(D) + '_d' + str(d) + '_N' + str(N) + '_L' + str(Ls[0]) + '_' + str(Ls[-1]) + '_Tm' + str(Tmax) + '.pdf')



